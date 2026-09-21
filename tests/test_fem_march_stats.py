"""``fem.stats["march"]`` — what each step of a march did, and ``jno.info(fem)`` reporting it.

The oracles are the inputs the user wrote (the τ grid, the continuation values, the time window) and
the march's own convergence test: a converged march has every recorded residual within its bound, a
failed one records the failing step at the index its error message names. Per-step wall time is
asserted only where a step is a host-visible event (continuation); a ``lax.scan`` march must say it has
none rather than invent one.
"""

from __future__ import annotations

import re

import jax
import numpy as np
import pytest

import jno

pytest.importorskip("meshio")

n = jno.np
grad, inner = n.grad, n.inner


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _load_path(nstep=12):
    """Nonlinear membrane under a load that ramps as τ^8: a march whose late steps work hardest."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(tau=(0.0, 1.0, nstep))
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X, tau = [co[0], co[1]], co[-1]
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols(value_shape=())
    return jno.fem([
        (1.0 + u * u) * inner(grad(u, X), grad(phi, X), 1) - 14.0 * tau**8 * phi + 0.0 * s.i(-1) * phi,
        s.evolves(s.i(-1)),
        u(*cb) - 0.0,
    ])


def _parametric():
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=6).domain()
    u, v = d.fem_symbols(names=("u", "v"))
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    k = n.parameter((1,), name="k")
    return jno.fem([(1.0 + k * ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 1.0 * vi, u(xb, yb) - 0.0])


def _heat(nonlinear, nt=21):
    d = jno.shape.rect(0, 0, 1, 1, size=0.25).domain(time=(0.0, 0.5, nt))
    x, y, t = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    c = d.variable("initial", split=True)
    u, v = d.fem_symbols()
    a, w = u.bind(x=x, y=y, t=t), v.bind(x=x, y=y, t=t)
    k = (1.0 + a * a) if nonlinear else 1.0
    return jno.fem([a.t * w + 0.1 * k * (a.x * w.x + a.y * w.y), u(b[0], b[1]) - 0.0,
                    u(c[0], c[1], c[2]) - n.sin(np.pi * c[0]) * n.sin(np.pi * c[1])])


CAPPED = dict(max_steps=1, rtol=1e-14, atol=1e-14)  # one Newton step at an unreachable tolerance


# --------------------------------------------------------------------------------------------------
# A load-path march: one compiled scan, judged per step.
# --------------------------------------------------------------------------------------------------
def test_load_path_records_every_step_against_its_bound():
    fem = _load_path(12)
    traj = np.asarray(fem.solve())
    m = fem.stats["march"]
    assert m["what"] == "load-path march" and m["coord"] == "τ"
    assert m["steps"] == traj.shape[0] == 12
    np.testing.assert_allclose(m["grid"], np.linspace(0.0, 1.0, 12))  # the grid the user declared
    assert m["residual"].shape == m["bound"].shape == (12,)
    assert np.all(m["residual"] <= m["bound"])  # converged: the march's own test, step by step
    assert m["step_s"] is None  # a scan step has no wall time of its own; none is invented

    text = str(jno.info(fem))
    assert "load-path march" in text and "all 12 steps converged" in text
    assert "only the mean" in text


def test_a_failed_load_path_records_the_step_its_error_names():
    fem = _load_path(12)
    fem.solve()
    with pytest.raises(RuntimeError, match="did not converge at step") as err:
        fem.solve(nonlinear=jno.solve.newton(**CAPPED))
    named = int(re.search(r"at step (\d+) of", str(err.value)).group(1))

    st = fem.stats
    assert st["error"].startswith("RuntimeError: fem.solve: the load-path march did not converge")
    assert st["solve_index"] == 2  # this failed solve, not the good one before it
    m = st["march"]
    first_bad = int(np.argmax(m["residual"] > m["bound"])) + 1  # 1-based, as the message counts
    assert first_bad == named
    assert f"FAILED at step {named} " in str(jno.info(fem))


# --------------------------------------------------------------------------------------------------
# Continuation: a Python loop whose steps ARE host-visible, so they get wall times.
# --------------------------------------------------------------------------------------------------
def test_continuation_records_per_step_time_and_residual():
    fem = _parametric()
    ks = np.linspace(0.0, 2.0, 5)
    fem.solve(continuation=jno.solve.continuation(k=ks))
    m = fem.stats["march"]
    assert m["what"] == "continuation" and m["coord"] == "k"
    np.testing.assert_allclose(m["grid"], ks)
    assert m["step_s"].shape == (5,) and np.all(m["step_s"] > 0)
    assert np.all(m["residual"] <= m["bound"])
    assert "time per step" in str(jno.info(fem))


def test_a_failed_continuation_still_records_the_refused_rung():
    fem = _parametric()
    with pytest.raises(RuntimeError, match=r"step \d+/4 .* did not converge") as err:
        fem.solve(nonlinear=jno.solve.newton(**CAPPED), continuation=jno.solve.continuation(k=np.linspace(0, 1, 4)))
    named = int(re.search(r"step (\d+)/4", str(err.value)).group(1))
    m = fem.stats["march"]
    assert m["steps"] == named  # every rung up to AND including the refused one ...
    assert m["residual"][-1] > m["bound"][-1]  # ... with the numbers that refused it
    assert np.all(m["residual"][:-1] <= m["bound"][:-1])  # the earlier rungs converged
    assert f"FAILED at step {named} " in str(jno.info(fem))


# --------------------------------------------------------------------------------------------------
# Transient: the solve returns a deferred node; the record fills in when it is evaluated.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("nonlinear", [False, True], ids=["linear", "nonlinear"])
def test_transient_record_is_deferred_until_evaluated(nonlinear):
    fem = _heat(nonlinear, nt=21)
    node = fem.solve()
    m = fem.stats["march"]
    assert m["deferred"] is True and m["steps"] == 20  # 21 time points = 20 steps
    assert m["window"] == (0.0, 0.5) and m["dt"] == pytest.approx(0.025)
    assert "not run yet" in str(jno.info(fem))

    traj = np.asarray(node.fn())
    assert traj.shape[0] == 21
    m = fem.stats["march"]
    assert m["deferred"] is False and m["evaluation"] == 1
    if nonlinear:  # judges its steps, so it has already synchronised: its time is real and free
        assert m["residual"].shape == (20,) and np.all(m["residual"] <= m["bound"])
        assert m["wall_s"] > 0
    else:  # never synchronised, and recording must not force it to: no time, and it says why
        assert m.get("residual") is None and m.get("wall_s") is None
        assert "not timed" in m["note"]
    node.fn()
    assert fem.stats["march"]["evaluation"] == 2
    text = str(jno.info(fem))
    assert ("evaluation 2" in text) if nonlinear else ("not timed" in text and "for the solve" not in text)


def test_evaluating_a_linear_transient_stays_asynchronous(monkeypatch):
    """Recording must never add a device sync: `.fn()` returns before the march has finished."""
    fem = _heat(False, nt=11)
    node = fem.solve()
    calls = []
    monkeypatch.setattr(jax, "block_until_ready", lambda x: calls.append(1) or x)
    node.fn()
    assert calls == []


def test_a_steady_solve_has_no_march_record():
    d = jno.shape.rect(0, 0, 1, 1, size=0.3).domain()
    x, y, _ = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    a, t = u.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem([a.x * t.x + a.y * t.y - 1.0 * t, u(b[0], b[1]) - 0.0])
    fem.solve()
    assert fem.stats["march"] is None
    assert "march" not in [s for s, _ in jno.info(fem).sections]


# --------------------------------------------------------------------------------------------------
# Reporting must not cost the solve anything it did not already pay.
# --------------------------------------------------------------------------------------------------
def _poisson():
    d = jno.shape.rect(0, 0, 1, 1, size=0.2).domain()
    x, y, _ = d.variable("interior", split=True)
    b = d.variable("boundary", split=True)
    u, v = d.fem_symbols()
    a, t = u.bind(x=x, y=y), v.bind(x=x, y=y)
    return jno.fem([a.x * t.x + a.y * t.y - 1.0 * t, u(b[0], b[1]) - 0.0])


def test_the_post_solve_line_does_no_matvec():
    """A residual matvec in the log line was 10 % of every GPU solve (6.7 of 63 ms at 73k dofs), and
    redundant with the solve's own residual gate. The line must work with no operator products."""
    fem = _poisson()
    out = fem.solve()

    class NoMatvec:
        def __matmul__(self, other):
            raise AssertionError("the post-solve log line computed a matvec")

    fem._A = NoMatvec()
    line = fem._solved_line(out, 0.1)
    assert "u in [0, " in line  # the range survived: nothing in the line touched the operator


def test_the_residual_is_reported_on_request():
    fem = _poisson()
    sol = fem.solve()
    b = np.asarray(fem._b)
    oracle = np.linalg.norm(np.asarray(fem._A @ sol) - b) / np.linalg.norm(b)
    row = dict(jno.info(sol, context=fem).sections[0][1])["rel. residual"]
    assert float(row.split()[0]) == pytest.approx(oracle, rel=1e-3)


def test_the_logging_wrapper_keeps_the_solve_signature_and_docs():
    """`solve` wraps `_solve_inner` for timing/logging; `(*args, **kwargs)` hid every solver slot."""
    import inspect

    from jno._fem import FEM

    assert inspect.signature(FEM.solve) == inspect.signature(FEM._solve_inner)
    assert {"linear", "precond", "nonlinear", "time", "adapt", "shard"} <= set(inspect.signature(FEM.solve).parameters)
    assert FEM.solve.__doc__ == FEM._solve_inner.__doc__ and FEM.solve.__name__ == "solve"
