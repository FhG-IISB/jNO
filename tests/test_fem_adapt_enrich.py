"""p-adaptivity — ``fem.solve(adapt=jno.solve.enrich(...))``.

Enrichment on a **fixed mesh**: the loop switches interpolation covers on at the marked nodes, so
the points and the connectivity are the same in every round and only the coefficient count per node
changes. That is what makes the assertions here sharper than an h-adaptive test's: with the geometry
held fixed there is no remesh to blur the comparison, so *which nodes were enriched* is an exact,
reproducible fact about the run rather than a statistic over a new mesh.

The oracles are accordingly structural rather than tolerances:

* an unenriched node's cover coefficients must be **exactly zero** — pinned, not merely small;
* enriching **every** node must reproduce the plain ``space="cover"`` solve to machine precision,
  which is the statement that the mask is the only thing the driver changes;
* a criterion peaked in a band must put the enrichment **in that band**, and a loop that enriched
  uniformly would pass a smoke test while being worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.utils.solver.fem_cover import cover_block

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True)
def _x64():
    """These oracles are exact identities (a mask reproducing a solve to machine precision), so they
    need float64 to mean anything -- 1e-9 is below float32 eps. The file used to inherit x64 from
    whichever sibling ran first in the same process, which is not a contract: under CI's one-process-
    per-file runner it ran in float32 and the identity test failed at 1.8e-07."""
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


BLK = cover_block(2)  # 3 in 2-D: the value slot plus the two cover coefficients


def _dense(a, b):
    """A direct solve. The enriched operator is gauged but ill-conditioned enough that the default
    matrix-free BiCGStab is the wrong tool for a 200-DOF test problem."""
    import jax.numpy as jnp

    return jnp.linalg.solve(jnp.asarray(a.todense() if hasattr(a, "todense") else a), jnp.asarray(b).reshape(-1))


def _grad_crit(ui):
    """The criterion the docs recommend: gradient magnitude, measured best per DOF of anything tried."""
    return jno.np.sqrt(ui.x**2 + ui.y**2 + 1e-30)


def _poisson(space="cover", size=0.25, rhs=None):
    """``-Δu = rhs`` with ``u = 0`` on the whole boundary of the unit square."""
    grad, inner = jno.np.grad, jno.np.inner
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    tol = 1e-9
    d.tag("walls", lambda *c: np.logical_or.reduce([(x < tol) | (x > 1 - tol) for x in c]))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols(space=space)
    body = inner(grad(u, X), grad(phi, X), 1)
    if rhs is not None:
        body = body - rhs(X) * phi
    fem = jno.fem([body, u(cw[0], cw[1]) - 0.0])
    fem._probe_trial = u.bind(x=co[0], y=co[1])  # for building a criterion in the tests
    return d, fem, X, co


def _sin_rhs(X):
    return 2 * np.pi**2 * jno.np.sin(np.pi * X[0]) * jno.np.sin(np.pi * X[1])


def _exact(pts):
    return np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])


def _l2(fem, d, u_vertex):
    """Relative error of the nodal values against the manufactured solution."""
    ex = _exact(np.asarray(d.mesh.points)[:, :2])
    return float(np.linalg.norm(np.asarray(u_vertex).reshape(-1) - ex) / np.linalg.norm(ex))


# ------------------------------------------------------------------ the mask is the whole mechanism


def test_an_unenriched_node_has_its_cover_coefficients_pinned_to_exactly_zero():
    """The definition of selective ``p``. Not "small": pinned — an unenriched node contributes the
    plain P1 hat and nothing else, which is what makes the blend across an order interface free."""
    d, fem, _X, _co = _poisson(rhs=_sin_rhs)
    fem.solve(_dense, adapt=jno.solve.enrich(criterion=_grad_crit(fem._probe_trial), theta=0.3, max_iters=2))

    mask = np.asarray(d._fem_enriched_nodes, dtype=bool)
    assert 0 < mask.sum() < mask.size, f"nothing to compare: {mask.sum()} of {mask.size} nodes enriched"
    pinned = {int(i) for i, _ in d._fem_native_dirichlet_pairs}
    for n in np.flatnonzero(~mask):
        assert {n * BLK + 1, n * BLK + 2} <= pinned, f"node {n} is unenriched but its covers are free"


def test_enriching_every_node_reproduces_the_plain_cover_solve():
    """The mask is the only thing the driver changes. With it all-true the space IS ``space='cover'``,
    so the two solves must agree to machine precision — a real oracle for the enriched path, since
    the plain one is verified independently against exact quadratics in test_fem_cover.py."""
    d0, fem0, _X, _co = _poisson(rhs=_sin_rhs)
    plain = np.asarray(fem0.solve(_dense)).reshape(-1)

    d1, fem1, _X, _co = _poisson(rhs=_sin_rhs)
    d1._fem_enriched_nodes = np.ones(len(d1.mesh.points), dtype=bool)
    full = np.asarray(jno.fem(fem1._constraints, **fem1._fem_kwargs).solve(_dense)).reshape(-1)

    assert np.abs(full - plain).max() < 1e-9, f"max |Δ| = {np.abs(full - plain).max():.3e}"


def test_the_returned_field_is_the_nodal_values_not_the_interleaved_block():
    """A cover field's DOFs interleave (value, cover, cover) per node, so the "first n_vert entries"
    convention that holds for P2 would hand back a mixture. Checked against the exact solution: a
    wrong slice is not merely inaccurate, it is not a solution field at all."""
    d, fem, _X, _co = _poisson(rhs=_sin_rhs)
    u = np.asarray(
        fem.solve(_dense, adapt=jno.solve.enrich(criterion=_grad_crit(fem._probe_trial), theta=0.9, max_iters=2))
    ).reshape(-1)
    assert u.size == np.asarray(d.mesh.points).shape[0], f"got {u.size} values for {len(d.mesh.points)} nodes"

    full = np.asarray(fem.solve(_dense)).reshape(-1)
    assert np.abs(u - full[::BLK]).max() < 1e-12, "the returned field is not the value slots"
    assert _l2(fem, d, u) < 0.05, f"relative error {_l2(fem, d, u):.3e} — that is not the solution"


# ------------------------------------------------------------------ it enriches where it is told


def test_a_criterion_enriches_where_it_peaks_and_leaves_the_rest_p1():
    """The headline, and the assertion an "it ran" test would miss: a criterion peaked on a diagonal
    ridge must concentrate the enrichment ON the ridge. The solution of this problem is smooth, so
    the recovery estimator would not choose these nodes."""
    d, fem, _X, co = _poisson(size=0.12, rhs=_sin_rhs)
    ridge = jno.np.exp(-(((co[0] + co[1] - 1.0) / 0.08) ** 2))
    fem.solve(_dense, adapt=jno.solve.enrich(criterion=ridge, theta=0.5, max_iters=3))

    mask = np.asarray(d._fem_enriched_nodes, dtype=bool)
    p = np.asarray(d.mesh.points)[:, :2]
    on = np.abs(p[:, 0] + p[:, 1] - 1.0) < 0.12
    assert on.any() and (~on).any()
    f_on, f_off = mask[on].mean(), mask[~on].mean()
    assert f_on > 2.0 * max(f_off, 1e-9), f"enrichment is not on the ridge (on {f_on:.2f}, off {f_off:.2f})"


def test_a_gradient_criterion_finds_the_reentrant_corner():
    """A gradient criterion on an L-shape must enrich the re-entrant corner -- where the gradient
    actually misbehaves. This replaces a test of the ZZ fallback, which no longer exists: recovery
    from the vertex values cannot see the cover coefficients, so it was removed rather than left as a
    default that reports a number anti-correlated with the error."""
    grad, inner = jno.np.grad, jno.np.inner
    s = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.14) - jno.Shape.rect(0.5, 0.5, 1.0, 1.0, size=0.14)
    d = s.domain()
    d.tag("walls", lambda x, y: np.ones_like(x, dtype=bool))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols(space="cover")
    fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) - 1.0 * phi, u(cw[0], cw[1]) - 0.0])
    fem.solve(_dense, adapt=jno.solve.enrich(criterion=_grad_crit(u.bind(x=X[0], y=X[1])), theta=0.4, max_iters=3))

    mask = np.asarray(d._fem_enriched_nodes, dtype=bool)
    p = np.asarray(d.mesh.points)[:, :2]
    r = np.hypot(p[:, 0] - 0.5, p[:, 1] - 0.5)
    near = r < 0.25
    assert mask[near].mean() > mask[~near].mean(), (
        f"the estimator missed the re-entrant corner (near {mask[near].mean():.2f}, far {mask[~near].mean():.2f})"
    )


def test_selective_enrichment_buys_accuracy_over_p1_at_a_fraction_of_the_full_cost():
    """The point of p-ADAPTIVITY rather than plain enrichment: better than P1 on the same mesh, and
    reached without paying for covers at every node. Both halves are asserted, because either alone
    is satisfied by something useless (enriching nothing; enriching everything)."""
    errs, dofs = {}, {}
    for tag in ("p1", "partial", "full"):
        d, fem, _X, _co = _poisson(space="Lagrange" if tag == "p1" else "cover", size=0.2, rhs=_sin_rhs)
        if tag == "partial":
            u = fem.solve(_dense, adapt=jno.solve.enrich(criterion=_grad_crit(fem._probe_trial), theta=0.3, max_iters=3))
            dofs[tag] = fem.adapt_history[-1]["n_dofs"]
            frac = float(np.asarray(d._fem_enriched_nodes).mean())
        else:
            sol = np.asarray(fem.solve(_dense)).reshape(-1)
            u = sol[:: BLK if tag == "full" else 1]
            pinned = len({int(i) for i, _ in d._fem_native_dirichlet_pairs})
            dofs[tag] = int(fem.dofs) - pinned
        errs[tag] = _l2(fem, d, u)

    assert 0.0 < frac < 1.0, f"the partial run enriched {frac:.0%} of the nodes — nothing to compare"
    assert errs["partial"] < errs["p1"], f"partial {errs['partial']:.3e} vs P1 {errs['p1']:.3e}"
    assert dofs["partial"] < dofs["full"], f"partial {dofs['partial']} DOFs vs full cover {dofs['full']}"


def test_the_caller_is_left_on_the_adapted_space_not_the_one_it_built():
    """After the loop, ``fem`` must BE the adapted problem — not just report it.

    The driver rebinds the caller's FEM to the final state, and the trap is that a rebinding of the
    operator used to be reverted by ``FEM.solve``'s own basis= bookkeeping on the way out: ``fem.A``
    was the adapted matrix while ``fem.solve()`` silently re-solved the space the caller had built,
    which here is EVERY node enriched. Same field, two different answers, nothing to say which.
    """
    d, fem, _X, _co = _poisson(size=0.2, rhs=_sin_rhs)
    u = np.asarray(
        fem.solve(_dense, adapt=jno.solve.enrich(criterion=_grad_crit(fem._probe_trial), theta=0.4, max_iters=3))
    ).reshape(-1)
    mask = np.asarray(d._fem_enriched_nodes, dtype=bool)
    assert 0 < mask.sum() < mask.size, "the run enriched everything or nothing — nothing to distinguish"

    again = np.asarray(fem.solve(_dense)).reshape(-1)
    assert np.abs(again[::BLK] - u).max() < 1e-12, "fem.solve() after the loop is not the adapted system"

    d2, fem2, _X2, _co2 = _poisson(size=0.2, rhs=_sin_rhs)
    d2._fem_enriched_nodes = mask.copy()
    ref = np.asarray(jno.fem(fem2._constraints, **fem2._fem_kwargs).solve(_dense)).reshape(-1)
    assert np.abs(again - ref).max() < 1e-9, "the final space is not the one the mask describes"


# ------------------------------------------------------------------ the loop's own bookkeeping


def test_the_history_records_the_enrichment_growing_round_by_round():
    """``adapt_history`` is the run's audit trail: ``n_enriched`` and the ACTIVE DOF count must both
    increase, since the padded total is constant and would hide the whole effect."""
    d, fem, _X, _co = _poisson(size=0.2, rhs=_sin_rhs)
    fem.solve(_dense, adapt=jno.solve.enrich(criterion=_grad_crit(fem._probe_trial), theta=0.3, max_iters=4))
    h = fem.adapt_history

    assert len(h) >= 3, f"only {len(h)} rounds recorded"
    assert h[0]["n_enriched"] == 0, "the loop must start at plain P1"
    enr = [r["n_enriched"] for r in h]
    act = [r["n_dofs"] for r in h]
    assert enr == sorted(enr) and enr[-1] > enr[0], f"enrichment did not grow: {enr}"
    assert act == sorted(act) and act[-1] > act[0], f"the active DOF count did not grow: {act}"
    assert all(r["enriched"].size == len(d.mesh.points) for r in h), "the recorded mask is not per node"


def test_the_dof_budget_stops_the_loop():
    """``max_dofs`` is a budget, and a budget that is only advisory is worse than none."""
    d, fem, _X, _co = _poisson(size=0.2, rhs=_sin_rhs)
    n_p1 = int(np.asarray(d.mesh.points).shape[0])
    fem.solve(
        _dense, adapt=jno.solve.enrich(criterion=_grad_crit(fem._probe_trial), theta=0.5, max_iters=8, max_dofs=n_p1 + 10)
    )
    h = fem.adapt_history

    assert len(h) < 8, f"max_dofs never fired: {len(h)} rounds"
    assert h[-1]["n_dofs"] >= n_p1 + 10, "the loop stopped for some other reason than the budget"
    assert all(r["n_dofs"] < n_p1 + 10 for r in h[:-1]), "the budget was passed before the last round"


@pytest.mark.parametrize("knob", ["tol", "eps"])
def test_a_stopping_rule_with_nothing_to_compare_against_is_refused(knob):
    """Neither knob has a meaning here, and a plausible-looking stop is worse than none.

    `estimate` is the norm of whatever drives the marking, and a criterion is a FIELD MAGNITUDE, not
    an error: measured on a plate problem the criterion norm ROSE 4.2967e+01 -> 4.3068e+01 over eight
    rounds while the true L2 error fell by a factor of three. `eps` reads that as a plateau and stops
    a converging loop; `tol` compares against a number unrelated to accuracy. Bound the run with
    `max_iters`/`max_dofs` instead."""
    _d, fem, _X, _co = _poisson(size=0.25, rhs=_sin_rhs)
    with pytest.raises(NotImplementedError, match="nothing to compare against"):
        fem.solve(_dense, adapt=jno.solve.enrich(criterion=_grad_crit(fem._probe_trial), max_iters=3, **{knob: 1e-3}))


def test_a_criterion_is_required():
    """No default estimator, by choice. ZZ recovery -- the obvious one -- reconstructs its gradient
    from the VERTEX VALUES, so it is blind to the cover coefficients enrichment adds and reports a
    number anti-correlated with the error (it rose 1.3353e-01 -> 1.3377e-01 while the true L2 fell
    4.692e-03 -> 2.677e-03). Guessing a criterion instead -- gradient of which field, reduced how? --
    would be a modelling choice made on the caller's behalf, so the loop asks."""
    _d, fem, _X, _co = _poisson(size=0.25, rhs=_sin_rhs)
    with pytest.raises(TypeError, match="criterion"):
        jno.solve.enrich(theta=0.5, max_iters=3)


# ------------------------------------------------------------------ it refuses what it cannot do


def test_a_field_without_covers_is_refused_by_name():
    """There is nothing to switch on in a Lagrange field, and enriching nothing while reporting a
    p-adaptive run is exactly the plausible-wrong-answer this codebase refuses to produce."""
    _d, fem, _X, _co = _poisson(space="Lagrange", rhs=_sin_rhs)
    with pytest.raises(ValueError, match="space='cover'"):
        fem.solve(_dense, adapt=jno.solve.enrich(criterion=_grad_crit(fem._probe_trial), max_iters=1))


def test_a_transient_problem_is_refused_by_name():
    """The state transfer that carries a solution across an adapt round is written for a change of
    MESH; a change of SPACE mid-march is not wired, and must say so rather than march on garbage."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(time=(0.0, 0.1, 5))
    u, phi = d.fem_symbols(space="cover")
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _tb = d.variable("boundary", split=True)
    xi0, yi0, _t0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [ui.t * vi + 0.1 * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(xi0, yi0) - 1.0],
    )
    with pytest.raises(NotImplementedError, match="steady adaptive loop"):
        fem.solve(adapt=jno.solve.enrich(criterion=_grad_crit(ui), max_iters=1))


def test_a_mask_built_on_a_different_mesh_is_refused():
    """The mask is one flag per mesh node. A stale one silently enriches the wrong nodes, which is
    the failure this guard exists to make loud."""
    d, fem, _X, _co = _poisson(rhs=_sin_rhs)
    d._fem_enriched_nodes = np.ones(len(d.mesh.points) + 3, dtype=bool)
    with pytest.raises(ValueError, match="enrichment mask"):
        jno.fem(fem._constraints, **fem._fem_kwargs).solve(_dense)
