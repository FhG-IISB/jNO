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
    assert h[0]["n_enriched"] == 0, "a FRESH run must start at plain P1"
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


def test_a_second_run_resumes_instead_of_starting_over():
    """Budgeted runs compose: `enrich` twice must continue, not throw the first run's covers away.

    The mask is the loop's state and it lives on the domain, so resuming is just reading it. Without
    this a second call restarts from P1 -- which is how h-then-p on an already-`space="cover"` field
    ended up REMOVING enrichment before adding it back, dropping the active DOF count at that call."""
    d, fem, _X, _co = _poisson(size=0.2, rhs=_sin_rhs)
    crit = _grad_crit(fem._probe_trial)
    fem.solve(_dense, adapt=jno.solve.enrich(criterion=crit, theta=0.4, max_iters=2))
    first = int(np.asarray(d._fem_enriched_nodes).sum())
    assert first > 0

    fem.solve(_dense, adapt=jno.solve.enrich(criterion=crit, theta=0.4, max_iters=2))
    second = int(np.asarray(d._fem_enriched_nodes).sum())
    assert fem.adapt_history[0]["n_enriched"] == first, (
        f"the second run restarted from {fem.adapt_history[0]['n_enriched']} instead of resuming {first}"
    )
    assert second > first, f"resuming must still add nodes: {first} -> {second}"


def test_a_vector_field_enriches_and_pins_every_component():
    """Covers on a VECTOR field. The layout is `(node*blk + slot)*vec + comp`, so an unenriched node
    has `vec` pins per cover slot, not one -- pinning by node index alone would leave every component
    but the first free, and the null space would come back with it."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.16).domain()
    tol = 1e-9
    d.tag("walls", lambda *c: np.logical_or.reduce([(x < tol) | (x > 1 - tol) for x in c]))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("walls", split=True)
    u, v = d.fem_symbols(value_shape=(2,), space="cover")
    a, t = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    exx, eyy, exy = a[0].x, a[1].y, 0.5 * (a[0].y + a[1].x)
    tr = exx + eyy
    sxx, syy, sxy = tr + 2 * exx, tr + 2 * eyy, 2 * exy
    f = jno.np.exp(-40.0 * ((xi - 0.3) ** 2 + (yi - 0.7) ** 2))
    fem = jno.fem([sxx * t[0].x + sxy * t[0].y + sxy * t[1].x + syy * t[1].y - f * t[1], u(xb, yb) - 0.0])
    crit = jno.np.sqrt(a[0].x ** 2 + a[0].y ** 2 + a[1].x ** 2 + a[1].y ** 2 + 1e-30)
    fem.solve(_dense, adapt=jno.solve.enrich(criterion=crit, theta=0.4, max_iters=2))

    mask = np.asarray(d._fem_enriched_nodes, dtype=bool)
    assert 0 < mask.sum() < mask.size, f"nothing to compare: {mask.sum()} of {mask.size} enriched"
    vec = 2
    pinned = {int(i) for i, _ in d._fem_native_dirichlet_pairs}
    for n in np.flatnonzero(~mask):
        want = {(n * BLK + slot) * vec + c for slot in range(1, BLK) for c in range(vec)}
        assert want <= pinned, f"node {n} is unenriched but {sorted(want - pinned)} are free"
    # An ENRICHED node keeps its covers free -- except along a Dirichlet wall, where the TANGENTIAL
    # one is not free to begin with: `u = 0` all along that wall fixes the tangential derivative, and
    # that is the mechanism making a cover field's trace the P1 interpolant of the data. The slot for
    # axis k is therefore pinned exactly when the node sits on a wall whose normal is some OTHER axis.
    pts = np.asarray(d.mesh.points)[:, :2]
    checked_wall = checked_interior = 0
    for n in np.flatnonzero(mask):
        on = [bool(abs(pts[n, k]) < 1e-9 or abs(pts[n, k] - 1.0) < 1e-9) for k in range(2)]
        checked_wall += any(on)
        checked_interior += not any(on)
        for k in range(2):
            want_pinned = any(on[j] for j in range(2) if j != k)
            got = {(n * BLK + 1 + k) * vec + c for c in range(vec)} & pinned
            assert bool(got) == want_pinned and len(got) in (0, vec), (
                f"node {n} at {pts[n]}: cover slot for axis {k} is "
                f"{'pinned' if got else 'free'}, expected {'pinned' if want_pinned else 'free'}"
            )
    assert checked_wall and checked_interior, (
        f"only one kind of enriched node was reached (wall {checked_wall}, interior {checked_interior})"
    )
    assert fem.adapt_history[-1]["n_dofs"] > fem.adapt_history[0]["n_dofs"]


def test_a_resumed_run_reuses_the_caller_s_operator_without_changing_the_answer(monkeypatch):
    """The opening rebuild is skipped when the caller's FEM already carries this exact mask, which is
    what a resumed run always hands it. A rebuild is the loop's dominant cost, so the saving is real --
    but only if it is a saving and not a shortcut, so the reused run is checked against one forced to
    rebuild: same enrichment, same field.

    The **enrichment mask** is compared exactly -- it is integer data and any difference there is a
    real defect. The **field** is compared to 1e-12 rather than bit-exactly, because bit-exactness is
    not available to assert on: the assembly's scatter-add is non-deterministic at the last bit on
    GPU, and this test measured 1.1e-16, 2.2e-16 and 3.3e-16 on three consecutive runs of identical
    code (it passes bit-exactly on CPU, which is what makes the source unambiguous). The tolerance is
    still four orders tighter than any real shortcut: a stale operator or a wrong mask moves the
    field by 1e-3 or more, not by an ULP."""

    def run(force_rebuild):
        d, fem, _X, _co = _poisson(size=0.25, rhs=_sin_rhs)
        crit = _grad_crit(fem._probe_trial)
        spec = lambda: jno.solve.enrich(criterion=crit, theta=0.4, max_iters=2)  # noqa: E731
        fem.solve(_dense, adapt=spec())
        if force_rebuild:
            d._fem_cover_mask_built = None  # the stamp is what proves reuse is safe; drop it
        calls = []
        real = jno.fem
        monkeypatch.setattr(jno, "fem", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
        out = np.asarray(fem.solve(_dense, adapt=spec())).reshape(-1)
        monkeypatch.undo()
        return len(calls), np.asarray(d._fem_enriched_nodes, dtype=bool), out

    n_reused, mask_reused, u_reused = run(force_rebuild=False)
    n_rebuilt, mask_rebuilt, u_rebuilt = run(force_rebuild=True)
    assert n_rebuilt == n_reused + 1 == 2, f"expected one build saved, got {n_reused} vs {n_rebuilt}"
    assert np.array_equal(mask_reused, mask_rebuilt), "reuse changed which nodes were enriched"
    np.testing.assert_allclose(
        u_reused,
        u_rebuilt,
        rtol=0.0,
        atol=1e-12,
        err_msg=f"reuse changed the field by {np.abs(u_reused - u_rebuilt).max():.3e}",
    )


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


# ------------------------------------------------------------------ 3-D, which was plumbed but unrun


def _poisson_3d(space, size=0.26):
    """``-Lap u = 3 pi^2 sin(pi x) sin(pi y) sin(pi z)`` on the unit cube, ``u = 0`` on every face.

    The manufactured solution vanishes on all six faces, so the homogeneous condition is EXACT and a
    cover field's inhomogeneous-trace limitation stays out of the measurement."""
    grad, inner = jno.np.grad, jno.np.inner
    d = jno.Shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=size).domain()
    tol = 1e-9
    d.tag("walls", lambda *c: np.logical_or.reduce([(x < tol) | (x > 1 - tol) for x in c]))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1], co[2]]
    u, phi = d.fem_symbols(space=space)
    sin = jno.np.sin
    f = 3 * np.pi**2 * sin(np.pi * X[0]) * sin(np.pi * X[1]) * sin(np.pi * X[2])
    stiff = inner(grad(u, X), grad(phi, X), 1)
    fem = jno.fem([stiff - f * phi, u(cw[0], cw[1], cw[2]) - 0.0])
    return d, fem, stiff, u.bind(x=X[0], y=X[1], z=X[2])


def _energy(fem, stiff, sol):
    """``1/2 integral |grad u_h|^2`` off the assembled form -- the SAME number for P1 and for an
    enriched space, which a nodal or geometric measure would not be."""
    return 0.5 * float(np.dot(np.asarray(sol).reshape(-1), np.asarray(fem.eval(stiff, sol)).reshape(-1)))


def test_enrichment_helps_in_3d_too():
    """3-D was plumbed -- ``cover_block(3)`` is 4 DOFs a node, and the mask strides by it -- but no
    test ran an assembled 3-D solve, let alone the loop. "It ran" is not evidence: the transient path
    marched happily while destroying its own initial condition.

    With a source and homogeneous data the Galerkin solution minimises ``J = 1/2 a(v,v) - (f,v)`` and
    ``J_h = -E_h``, so the energy RISES toward the truth: more energy is a better answer, and that is
    the oracle. P1, full enrichment and a p-adaptive run are ordered by it."""
    d1, fem1, st1, _ui = _poisson_3d("Lagrange")
    e_p1 = _energy(fem1, st1, _dense(fem1.A, fem1.b))

    d2, fem2, st2, _ = _poisson_3d("cover")
    e_full = _energy(fem2, st2, _dense(fem2.A, fem2.b))

    d3, fem3, st3, ui3 = _poisson_3d("cover")
    crit = jno.np.sqrt(ui3.x**2 + ui3.y**2 + ui3.z**2 + 1e-30)
    fem3.solve(_dense, adapt=jno.solve.enrich(criterion=crit, theta=0.5, max_iters=3))
    e_adapt = _energy(fem3, st3, _dense(fem3.A, fem3.b))
    frac = float(np.asarray(d3._fem_enriched_nodes).mean())

    assert cover_block(3) == 4, "a 3-D cover node carries its value plus three cover coefficients"
    assert e_full > e_p1, f"enrichment must add energy in 3-D: full {e_full:.6f} vs P1 {e_p1:.6f}"
    assert e_adapt > e_p1, f"the p-adaptive run must beat P1: {e_adapt:.6f} vs {e_p1:.6f}"
    assert e_adapt <= e_full + 1e-12, "a partial enrichment cannot beat enriching everything"
    assert 0.0 < frac < 1.0, f"nothing was chosen in 3-D (enriched {frac:.0%})"
