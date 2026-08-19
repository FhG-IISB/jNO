"""h-adaptivity on a COUPLED problem, driven by a criterion written in the traced DSL.

The h path assumed a single trial function in two independent places, so every multifield problem --
every velocity/pressure saddle, i.e. every incompressible flow -- was out of reach. One of the two
assumptions refused loudly; the other returned zeros and reported a clean run having refined nothing.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytest.importorskip("mmgpy", reason="mmgpy required for adaptive remeshing")

import jno

meshio = pytest.importorskip("meshio")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _stokes(size=0.4):
    """Taylor-Hood channel: fields are [u (vector, P2), p (scalar, P1)] -- p is NOT field 0."""
    d = jno.Shape.rect(0.0, 0.0, 3.0, 1.0, size=size).domain()
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    x, y, _ = d.variable("interior", split=True)
    cin = d.variable("inlet", where=lambda X, Y: X < 1e-9, split=True)
    cbot = d.variable("bottom", where=lambda X, Y: Y < 1e-9, split=True)
    eu, ev = jno.np.symgrad(u, [x, y]), jno.np.symgrad(v, [x, y])
    dd = lambda a, b: jno.np.inner(a, b, n_contract=2)  # noqa: E731
    pp, qq = p.bind(x=x, y=y), q.bind(x=x, y=y)
    fem = jno.fem(
        [
            2.0 * dd(eu, ev) - pp * jno.np.trace(ev),
            -qq * jno.np.trace(eu),
            u(cin[0], cin[1])[0] - 1.0,
            u(cin[0], cin[1])[1] - 0.0,
            u(cbot[0], cbot[1])[0] - 0.0,
            u(cbot[0], cbot[1])[1] - 0.0,
            p.pin(),
        ],
        quad_degree=3,
    )
    return d, fem, dd(eu, eu)  # the criterion: shear-rate magnitude, written by the caller


def test_a_traced_criterion_refines_a_coupled_problem():
    """`_vertex_view` was taken unconditionally, one line before the branch that uses it -- and a
    `criterion=` replaces the recovery estimator entirely and reads the FULL solution vector, so that
    view is never consulted on this path. Every multifield problem was refused on a value the branch
    does not need."""
    d, fem, shear = _stokes()
    n0 = int(np.asarray(d.mesh.points).shape[0])
    fem.solve(adapt=jno.solve.remesh(criterion=shear, metric_field=1, max_iters=3, theta=0.5))
    n1 = int(np.asarray(fem.domain.mesh.points).shape[0])
    assert n1 > n0, f"the mesh was never refined: {n0} -> {n1}"


def test_a_criterion_on_a_non_first_field_is_not_silently_zero():
    """The failure this exists for. `_criterion_indicators` read `num[:n_vert]`, which assumes the
    tested field starts at DOF 0 -- true only for field 0. Pointed at the pressure block it read the
    untouched velocity rows, so every indicator came back exactly zero: nothing marked, `estimate`
    0.0, and the driver reported a clean run having refined nothing. A silent no-op, not a raise."""
    d, fem, shear = _stokes()
    fem.solve(adapt=jno.solve.remesh(criterion=shear, metric_field=1, max_iters=2, theta=0.5))
    hist = fem.adapt_history
    assert hist, "no adaptive rounds ran"
    assert hist[0]["estimate"] > 0.0, f"the criterion assembled to zero: {hist[0]}"
    assert hist[0]["n_marked"] > 0, f"nothing was marked despite a non-zero criterion: {hist[0]}"


def test_cell_size_is_usable_in_a_criterion():
    """`dom.cell_size` is a geometry symbol resolved from the domain context, not a region. The
    criterion's region resolver counted its tag as a region and looked for one called 'cell_size'."""
    d = jno.Shape.rect(0.0, 0.0, 2.0, 1.0, size=0.4).domain()
    u, phi = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0], quad_degree=3)
    n0 = int(np.asarray(d.mesh.points).shape[0])
    fem.solve(adapt=jno.solve.remesh(criterion=d.cell_size * ui * ui, max_iters=2, theta=0.6))
    assert int(np.asarray(fem.domain.mesh.points).shape[0]) > n0


def test_a_single_field_problem_still_refines_on_the_zz_estimator():
    """The vertex view is now taken lazily; the estimator path must be untouched by that."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    u, phi = d.fem_symbols()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=x, y=y), phi.bind(x=x, y=y)
    f = jno.np.exp(-40.0 * ((x - 0.62) ** 2 + (y - 0.35) ** 2))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    n0 = int(np.asarray(d.mesh.points).shape[0])
    fem.solve(adapt=jno.solve.remesh(max_iters=2, theta=0.5))  # no criterion -> ZZ
    assert int(np.asarray(fem.domain.mesh.points).shape[0]) > n0
