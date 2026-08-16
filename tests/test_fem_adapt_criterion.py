"""Refining on a **traced criterion** instead of the recovery error estimator.

The adaptive loop could only mark on ZZ recovery or a Hessian metric. Production AMR codes mark on
*physical* quantities — a density gradient, an interface, vorticity, a shock detector — and all of
those are already expressible in jNO's language; there was simply no way to hand one to the loop.

A criterion carries **no test function**: it is a field, not an equation. One is supplied internally,
because ``FEM.eval`` assembles weak terms and refuses anything without a test function — and it has to
be the problem's *own* test symbol, since a fresh ``domain.fem_symbols()`` makes a different field
that the assembler rejects outright.

The assertions here are about **where the mesh ends up**, not about the plumbing running: a criterion
that returns numbers and refines uniformly would pass any smoke test and be worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno
from jno.utils.solver.fem_adapt import _criterion_indicators, _element_gradients, zz_error_indicators

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _ridge_problem(cell="simplex", size=0.09):
    """-Delta u = a thin diagonal ridge source. Returns the pieces a criterion is written from."""
    s = jno.Shape.rect(0, 0, 1, 1, size=size)
    d = (s.quad() if cell == "tensor" else s).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    f = jno.np.exp(-(((xi + yi - 1.0) / 0.06) ** 2))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0])
    return d, fem, ui, vi, xi, yi


def _size_on_and_off_ridge(dom, halfwidth=0.10):
    """Mean cell size on the ridge and away from it — where the DOFs actually went."""
    p = np.asarray(dom.mesh.points)[:, :2]
    _, meas, cells = _element_gradients(dom)
    h = np.sqrt(meas)
    on = np.abs(p[cells].mean(axis=1).sum(axis=1) - 1.0) < halfwidth
    return h[on].mean(), h[~on].mean()


# ------------------------------------------------------------------ it refines where the criterion says


@pytest.mark.parametrize("cell", ["simplex", "tensor"])
def test_a_coordinate_criterion_refines_where_it_peaks(cell):
    """The headline. A criterion peaked on the ridge must put the small cells there — and the ridge is
    NOT where the ZZ estimator would refine, since the solution of this problem is smooth."""
    d, fem, _ui, _vi, xi, yi = _ridge_problem(cell)
    crit = jno.np.exp(-(((xi + yi - 1.0) / 0.06) ** 2))
    fem.solve(adapt=jno.solve.remesh(criterion=crit, theta=0.4, max_iters=3, refine_factor=2.0))
    on, off = _size_on_and_off_ridge(fem.domain)
    assert off / on > 1.5, f"{cell}: cells are not concentrated on the ridge (off/on = {off / on:.2f})"


def test_a_criterion_needs_no_test_function_but_accepts_one():
    """Both spellings assemble to the same indicators: the bare expression is the contract, and a term
    that already carries the test function is accepted rather than refused."""
    _d, fem, ui, vi, _xi, _yi = _ridge_problem()
    sol = np.asarray(fem.solve()).reshape(-1)
    bare, _ = _criterion_indicators(fem, jno.np.abs(ui.x), sol)
    weak, _ = _criterion_indicators(fem, jno.np.abs(ui.x) * vi, sol)
    np.testing.assert_allclose(bare, weak, rtol=1e-6)


def test_a_criterion_may_use_freshly_fetched_coordinates():
    """`jno.fem` retags a form's coordinates to the quadrature pool at build time. A criterion is given
    LATER, so coordinates fetched fresh from the domain still pointed at the mesh pool — and the
    mismatch surfaced as a bare broadcasting error ('input type=float32[173484] and requested type=
    float32[948]') naming neither the criterion nor the coordinates. Both spellings must agree."""
    d, fem, _ui, _vi, xi, yi = _ridge_problem()
    sol = np.asarray(fem.solve()).reshape(-1)
    xf, yf, _ = d.variable("interior", split=True)
    own, _ = _criterion_indicators(fem, jno.np.exp(-(((xi + yi - 1.0) / 0.06) ** 2)), sol)
    fresh, _ = _criterion_indicators(fem, jno.np.exp(-(((xf + yf - 1.0) / 0.06) ** 2)), sol)
    np.testing.assert_allclose(own, fresh, rtol=1e-6)


def test_the_indicator_scale_is_the_criterion_not_the_cell_volume():
    """`FEM.eval` gives `int g phi_i`; dividing by the lumped mass makes the nodal field `g` rather than
    `g x volume`, so a criterion means the same thing on a coarse and a fine mesh. Checked by halving
    the mesh size and asserting the indicator's SCALE is unchanged (its per-cell integral falls with
    the cell, but the field it integrates does not)."""
    vals = []
    for size in (0.12, 0.06):
        _d, fem, _ui, _vi, xi, yi = _ridge_problem(size=size)
        sol = np.asarray(fem.solve()).reshape(-1)
        eta, _ = _criterion_indicators(fem, 1.0 + 0.0 * xi, sol)  # criterion == 1 everywhere
        vals.append(eta.sum())  # sum of int_K 1 = the domain area, mesh-independent
    np.testing.assert_allclose(vals, 1.0, rtol=1e-6)


def test_a_constant_criterion_marks_by_cell_size_alone():
    """The degenerate case, worth pinning: with `criterion=1` the indicator is the cell volume, so
    Dorfler marks the biggest cells and the loop performs uniform-ish refinement rather than doing
    something undefined."""
    _d, fem, _ui, _vi, xi, _yi = _ridge_problem()
    n0 = len(fem.domain.mesh.points)
    fem.solve(adapt=jno.solve.remesh(criterion=1.0 + 0.0 * xi, theta=0.6, max_iters=2))
    assert len(fem.domain.mesh.points) > n0


# ------------------------------------------------------------------------------ it is a real choice


def test_the_criterion_replaces_the_estimator_rather_than_blending_with_it():
    """A criterion equal to nothing the estimator sees must still drive the mesh — otherwise the loop
    is quietly still following ZZ. Compared against the ZZ marking on the same solution: the two
    indicator fields must disagree, and the criterion's must be the one that peaks on the ridge."""
    d, fem, _ui, _vi, xi, yi = _ridge_problem()
    sol = np.asarray(fem.solve()).reshape(-1)
    crit_eta, _ = _criterion_indicators(fem, jno.np.exp(-(((xi + yi - 1.0) / 0.06) ** 2)), sol)
    zz_eta, _ = zz_error_indicators(d, sol)

    p = np.asarray(d.mesh.points)[:, :2]
    _, _meas, cells = _element_gradients(d)
    on_ridge = np.abs(p[cells].mean(axis=1).sum(axis=1) - 1.0) < 0.10
    # the criterion's worst cells are on the ridge; ZZ's are not concentrated there in the same way
    assert on_ridge[np.argmax(crit_eta)], "the criterion did not peak on the ridge"
    assert crit_eta[on_ridge].mean() / crit_eta[~on_ridge].mean() > 5.0
    assert not np.allclose(crit_eta / crit_eta.max(), zz_eta / max(zz_eta.max(), 1e-30))


# --------------------------------------------------------------------------------------- refusals


def test_a_criterion_with_no_coordinates_is_refused():
    """Without coordinates its region is undetermined, and guessing would silently integrate it over
    the wrong part of the mesh."""
    _d, fem, _ui, _vi, _xi, _yi = _ridge_problem()
    sol = np.asarray(fem.solve()).reshape(-1)
    with pytest.raises(ValueError, match="carries no coordinates"):
        _criterion_indicators(fem, jno.np.abs(1.0), sol)


def test_an_out_of_range_metric_field_is_refused():
    _d, fem, ui, _vi, _xi, _yi = _ridge_problem()
    sol = np.asarray(fem.solve()).reshape(-1)
    with pytest.raises(ValueError, match="metric_field"):
        _criterion_indicators(fem, jno.np.abs(ui.x), sol, field=3)


def test_the_criterion_reaches_the_public_slot():
    assert jno.solve.remesh().criterion is None
    spec = jno.solve.remesh(criterion=42)
    assert spec.criterion == 42
