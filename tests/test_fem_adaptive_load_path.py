"""Adaptive load stepping — ``fem.solve(tau=jno.solve.adaptive(limit=...))`` on a ``domain(tau=...)`` march.

A uniform load grid is wrong in both directions at once: it wastes steps while nothing happens, and
takes too-large ones through the event. On a **path-dependent** march (history + irreversibility) the
second is not merely coarse — a step can converge perfectly and skip the whole event, giving a valid
sequence of equilibria with no resolved transition between them, which is a *different* answer.

The criterion cannot be the transient's. A rate-independent load path has no local truncation error to
estimate: each step is an equilibrium, not an approximation to a trajectory, so a Richardson
step-doubling estimate measures nothing. ``limit`` bounds how much the solution may change in one step.

Mechanism: **pilot → freeze → replay**. March eagerly with rejection to discover an accepted schedule,
freeze it, replay as a fixed-length differentiable scan. Rejection is why the pilot must be separate —
the transient marcher accepts every attempt on purpose (a discarded state makes the per-step adjoint run
at zero cotangent and returns NaN), and the replay has nothing to reject.

Oracles:
* **the steps go where the action is** — on a load that arrives almost entirely at the end
  (``P ∝ τ⁸``), the accepted steps must cluster there, and every accepted step must respect ``limit``.
* **the limit is what a uniform grid violates** — the same problem on the declared uniform grid takes
  per-step jumps far over ``limit``; that is the failure adaptivity exists to remove.
* **accuracy at fewer steps** — the adaptive answer matches a much finer uniform reference.
* **replay reproduces the pilot** — the frozen schedule, replayed as a scan, gives the pilot's answer.
* **differentiability** — FD-checked through the replay.
* **fail-loud** — missing ``limit``, ``limit`` in the wrong slot, ``tau=`` without a march, and the
  unstable-branch floor (which no amount of step cutting can fix, and which says so).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jno

LIMIT = 0.02


def _aliases():
    n = jno.np
    return n.grad, n.inner


def _burst_march(nstep, *, power=8.0, peak=14.0, size=0.25):
    """A linear elastic membrane whose load arrives almost entirely at the end of the path:
    ``P(τ) = peak·τ^power``. The response is ``u ∝ τ^power``, so a uniform τ grid takes tiny steps
    while nothing happens and a huge one through the finish — the shape adaptivity is for.

    An inert state field carries no physics; it is there because ``.i(k)`` is what triggers the march.
    """
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(tau=(0.0, 1.0, nstep))
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X, tau = [co[0], co[1]], co[-1]
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols(value_shape=())
    return jno.fem(
        [
            inner(grad(u, X), grad(phi, X), 1) - peak * (tau**power) * phi + 0.0 * s.i(-1) * phi,
            s.evolves(s.i(-1)),
            u(*cb) - 0.0,
        ]
    )


# --------------------------------------------------------------------------------------------------
# Oracle 1 — the steps go where the action is, and every accepted step respects the limit.
# --------------------------------------------------------------------------------------------------
def test_the_schedule_concentrates_steps_on_the_event():
    fem = _burst_march(9)
    traj = np.asarray(fem.solve(tau=jno.solve.adaptive(limit=LIMIT)))
    sched = np.asarray(fem._tau_schedule)

    assert sched[0] == 0.0 and abs(sched[-1] - 1.0) < 1e-12, "the schedule must span the declared path"
    assert np.all(np.diff(sched) > 0), "the schedule must be increasing"
    dt = np.diff(sched)
    assert dt[-1] < 0.3 * dt[0], f"steps did not shrink towards the event: first {dt[0]:.4g}, last {dt[-1]:.4g}"
    # More than half the steps land in the last quarter of the path, where the load actually arrives.
    assert (sched[1:] > 0.75).sum() > 0.5 * len(dt), "the steps did not cluster where the load arrives"

    # The output keeps the declared shape — the step size is decoupled from the sample times, exactly as
    # the transient stepper resamples onto its save grid.
    assert traj.shape[0] == 9


def test_every_accepted_step_respects_the_limit():
    """Measured on the SCHEDULE, not the resampled output: that is what the controller bounds."""
    fem = _burst_march(9)
    fem.solve(tau=jno.solve.adaptive(limit=LIMIT))
    sched = np.asarray(fem._tau_schedule)
    # u ∝ τ⁸ pointwise, so the per-step change is the peak displacement times Δ(τ⁸).
    fine = np.asarray(_burst_march(2).solve())  # τ = 0 and 1: the peak response
    peak = np.abs(fine[-1]).max()
    jumps = peak * np.abs(np.diff(sched**8.0))
    assert jumps.max() <= LIMIT * 1.05, f"an accepted step moved {jumps.max():.4g}, over the limit {LIMIT}"


# --------------------------------------------------------------------------------------------------
# Oracle 2 — the control: the declared UNIFORM grid violates the same limit badly. Without this, oracle
# 1 could be satisfied by a grid that never needed adapting.
# --------------------------------------------------------------------------------------------------
def test_the_uniform_grid_violates_the_limit_the_adaptive_one_holds():
    fem = _burst_march(9)
    uniform = np.asarray(fem.solve())  # no tau= : the declared grid
    jumps = np.abs(np.diff(uniform, axis=0)).max(axis=1)
    assert jumps.max() > 5 * LIMIT, f"the uniform grid is already fine enough ({jumps.max():.4g}) — no test here"

    fem2 = _burst_march(9)
    fem2.solve(tau=jno.solve.adaptive(limit=LIMIT))
    assert len(fem2._tau_schedule) > len(uniform)


# --------------------------------------------------------------------------------------------------
# Oracle 3 — accuracy: the adaptive answer matches a much finer uniform reference. The load is only a
# function of τ here, so the endpoint is exact either way; the *path* is what adaptivity gets right, so
# compare the whole resampled trajectory.
# --------------------------------------------------------------------------------------------------
def test_adaptive_matches_a_fine_uniform_reference():
    n_out = 9
    adaptive = np.asarray(_burst_march(n_out).solve(tau=jno.solve.adaptive(limit=LIMIT)))
    fine = np.asarray(_burst_march(2001).solve())
    # Sample the fine march at the same τ values the coarse output reports.
    idx = np.searchsorted(np.linspace(0.0, 1.0, 2001), np.linspace(0.0, 1.0, n_out))
    ref = fine[np.clip(idx, 0, 2000)]
    assert np.abs(ref).max() > 1e-3
    # The output is LINEARLY INTERPOLATED from the schedule onto the declared grid (the same contract
    # the transient stepper's `save_ts` resampling makes), so the bound is `limit` itself -- which is
    # precisely the accuracy the caller asked for. Machine precision would be the wrong assertion.
    err = np.abs(adaptive - ref).max()
    assert err <= LIMIT, f"adaptive is outside its own limit against the fine reference: {err:.3e}"


# --------------------------------------------------------------------------------------------------
# Oracle 4 — replay reproduces the pilot. The whole differentiability contract rests on the frozen
# schedule being replayable, so the two legs must give the same trajectory.
# --------------------------------------------------------------------------------------------------
def test_replay_over_the_frozen_schedule_reproduces_the_pilot():
    from jno.utils.solver.history_march import run_history_march

    fem = _burst_march(9)
    adaptive = np.asarray(fem.solve(tau=jno.solve.adaptive(limit=LIMIT)))
    sched = np.asarray(fem._tau_schedule)

    # Replay the SAME schedule through the ordinary fixed-grid march by declaring it as the grid.
    fem_replay = _burst_march(len(sched))
    fem_replay.domain._time_points = sched  # the frozen schedule, marched as if it had been declared
    raw = np.asarray(run_history_march(fem_replay))
    resampled = np.asarray(
        jax.vmap(lambda c: jnp.interp(jnp.linspace(0.0, 1.0, 9), jnp.asarray(sched), c), in_axes=1, out_axes=1)(
            jnp.asarray(raw)
        )
    )
    assert np.abs(adaptive - resampled).max() < 1e-10, "the replay did not reproduce the pilot's path"


# --------------------------------------------------------------------------------------------------
# Oracle 5 — the two-step workflow, and differentiability through the replayed schedule.
#
# A pilot cannot run on a parametric form: it needs concrete values to accept or reject a step and a
# differentiable solve hands it tracers. Piloting at the parameters' STORED values would be the obvious
# fallback and is a silent trap — a fresh `jno.np.parameter` stores 0.0, so the schedule would be
# adapted to a load path with the parameter switched off. So that combination refuses by name, and the
# supported route is: discover the schedule on the forward study, replay it for the gradient.
# --------------------------------------------------------------------------------------------------
def _parametric_burst(nstep=5, size=0.3):
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(tau=(0.0, 1.0, nstep))
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X, tau = [co[0], co[1]], co[-1]
    kP = jno.np.reshape(jno.np.parameter((1,), name="pk"), ())
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols(value_shape=())
    return jno.fem(
        [
            inner(grad(u, X), grad(phi, X), 1) + 0.1 * u * u * phi - 30.0 * kP * (tau**4.0) * phi + 0.0 * s.i(-1) * phi,
            s.evolves(s.i(-1)),
            u(*cb) - 0.0,
        ]
    )


def test_piloting_a_parametric_march_refuses_and_names_the_workflow():
    fem = _parametric_burst()
    with pytest.raises(NotImplementedError, match="tau_schedule|runtime parameter"):
        fem.solve(tau=jno.solve.adaptive(limit=0.01))


def test_gradient_flows_through_a_replayed_schedule():
    # Step 1: discover a schedule on the forward (non-parametric) study of the same shape.
    fwd = _burst_march(5, power=4.0, peak=30.0, size=0.3)
    fwd.solve(tau=jno.solve.adaptive(limit=0.01))
    schedule = fwd.tau_schedule
    assert len(schedule) > 5, "the forward pilot did not adapt — the replay would prove nothing"

    # Step 2: replay it on the parametric form. Same schedule, now differentiable.
    fem = _parametric_burst()
    node = fem.solve(tau=schedule)
    assert type(node).__name__ == "FunctionCall", "a parametric replay must stay a trace node"
    assert np.array_equal(fem.tau_schedule, schedule)

    fn = node.fn

    def endpoint(kv):
        return jnp.sum(jnp.asarray(fn(jnp.reshape(kv, (1,)))[-1] ** 2))

    g = jax.grad(endpoint)(1.0)
    fd = (endpoint(1.0 + 1e-6) - endpoint(1.0 - 1e-6)) / 2e-6
    assert np.isfinite(g) and abs(g) > 0, "the gradient vanished through the replayed schedule"
    assert np.allclose(g, fd, rtol=1e-6), f"AD {g:.8e} vs FD {fd:.8e}"


def test_an_explicit_schedule_must_span_the_declared_path():
    fem = _burst_march(5)
    with pytest.raises(ValueError, match="must agree at both ends"):
        fem.solve(tau=np.array([0.0, 0.5, 0.9]))
    with pytest.raises(ValueError, match="strictly increasing"):
        fem.solve(tau=np.array([0.0, 0.6, 0.3, 1.0]))


# --------------------------------------------------------------------------------------------------
# Oracle 6 — fail loud.
# --------------------------------------------------------------------------------------------------
def test_adaptive_without_a_limit_fails_loud():
    fem = _burst_march(5)
    with pytest.raises(ValueError, match="limit"):
        fem.solve(tau=jno.solve.adaptive())


def test_limit_in_the_time_slot_fails_loud():
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain(time=(0.0, 0.1, 5))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    u, phi = d.fem_symbols(names=("tu", "tphi"))
    ui, pi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * pi + (ui.x * pi.x + ui.y * pi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0])
    with pytest.raises(ValueError, match="limit"):
        fem.solve(time=jno.solve.adaptive(limit=0.1))


def test_tau_slot_without_a_march_fails_loud():
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.4).domain()
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) - 1.0 * phi, u(*cb) - 0.0])
    with pytest.raises(ValueError, match="load-path|does not march"):
        fem.solve(tau=jno.solve.adaptive(limit=0.1))


def test_an_unreachable_limit_names_the_unstable_branch():
    """A limit no step size can meet is the snap-back signature, and step cutting cannot fix it. The
    message has to say that rather than suggesting a smaller step."""
    fem = _burst_march(5, power=1.0, peak=200.0)
    with pytest.raises(RuntimeError, match="UNSTABLE|snap-back|arc-length"):
        fem.solve(tau=jno.solve.adaptive(limit=1e-7))


def test_a_slack_limit_keeps_the_declared_grid_cost():
    """A limit nothing violates must not make the march expensive — the controller should grow back."""
    fem = _burst_march(9, power=1.0, peak=0.01)
    fem.solve(tau=jno.solve.adaptive(limit=10.0))
    assert len(fem._tau_schedule) <= 10, f"a slack limit still took {len(fem._tau_schedule)} steps"


def test_a_shrinking_step_does_not_make_acceptance_harder():
    """The pilot's accept test must not tighten as the step shrinks.

    It used to be a pure residual REDUCTION, ``r_after <= 1e-6 * r_before``. Shrinking a step leaves
    the previous state closer to equilibrium at the new τ, so ``r_before`` falls with ``dtau`` and the
    bar falls with it — every cut made acceptance strictly HARDER, so a step rejected once for any
    reason spiralled to the floor and reported an "unstable branch" whatever the true cause was.
    Observed on the 3-D SENT march: it died at τ=0 with the LIMIT satisfied (overshoot ×0).

    Driving with a deliberately tiny `dt0` puts every step in exactly that regime — the previous state
    is already nearly in equilibrium at the next τ — so this march is unsolvable under the old rule and
    routine under the new one.
    """
    fem = _burst_march(6)
    # dt0 tiny => the previous state is already nearly in equilibrium at the next τ on EVERY step, which
    # is precisely the regime the old reduction test could not pass.
    sol = np.asarray(fem.solve(tau=jno.solve.adaptive(limit=0.5, dt0=1e-6, max_steps=400)))
    assert np.all(np.isfinite(sol))
    ref = np.asarray(_burst_march(6).solve())
    # Compare at τ = 1: the schedule ends exactly there, so that frame is a solved node rather than an
    # interpolation onto the declared grid (interior frames carry resampling error bounded by `limit`,
    # which this file documents elsewhere and which is not what this test is about).
    assert np.abs(ref[-1]).max() > 1e-3, "the reference is trivial — the comparison would be vacuous"
    assert np.abs(sol[-1] - ref[-1]).max() / np.abs(ref[-1]).max() < 1e-6, "the adaptive path moved the answer"
    sched = np.asarray(fem.tau_schedule)
    assert sched[0] == 0.0 and abs(sched[-1] - 1.0) < 1e-9, "the schedule must span the declared path"
    assert len(sched) > 2, "the pilot never took a step"


def test_the_pilot_scores_the_min_map_on_a_bounded_march():
    """The pilot judged a trial step by ``op.residual``. Under ``field.bounds(lo, hi)`` that is the wrong
    function: the driver root-finds the min-map, and on an ACTIVE bound the bare residual is non-zero by
    construction (it is the multiplier), so it never falls however well the step solved. Every step then
    looked unconverged — and since a rejection only shrinks the step, which cannot release a bound, the
    pilot cut to its floor and reported an UNSTABLE BRANCH. Measured on the 3-D SENT march: it died at
    τ=0 with the limit satisfied (overshoot ×0).

    Here the bound is genuinely active (the source drives the field well past the cap), so the pilot
    cannot finish at all unless it scores the right function.
    """
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(tau=(0.0, 1.0, 5))
    d.tag("left", lambda x, y: x < 1e-9)
    co, cl = d.variable("interior", split=True), d.variable("left", split=True)
    X, tau = [co[0], co[1]], co[-1]
    u, phi = d.fem_symbols()
    s, _ = d.fem_symbols(value_shape=())
    fem = jno.fem(
        [
            inner(grad(u, X), grad(phi, X), 1) + 10.0 * (u - 5.0 * tau) * phi + 0.0 * s.i(-1) * phi,
            s.evolves(u),
            u.bounds(0.0, 1.0),
            u(*cl) - 0.0,
        ]
    )
    traj = np.asarray(fem.solve(tau=jno.solve.adaptive(limit=0.25, max_steps=300)))
    sched = np.asarray(fem.tau_schedule)
    assert abs(sched[-1] - 1.0) < 1e-9, "the pilot never reached the end of the path"
    assert traj.max() <= 1.0 + 1e-9, "the upper bound must hold"
    assert traj.max() > 1.0 - 1e-6, "the bound must actually be ACTIVE, or this tests nothing"
