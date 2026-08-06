"""Mesh motion written as a **term**: ``coord.d(t) - velocity`` in the ``jno.fem([...])`` list.

A coordinate is one of exactly three things, and each is an existing spelling — no new method:

===============  ==========================  ==========================================
a coordinate is  you write                   who moves it
===============  ==========================  ==========================================
fixed            nothing                     nobody
free             ``coord.trainable()``       an optimiser, or ``jno.solve.relocate()``
determined       ``coord.d(t) - v`` (a term) the march
===============  ==========================  ==========================================

These cases pin down the **classification**: which residuals are geometry terms, which are emphatically
not, and that nothing about it is boundary-specific — an interior region, a boundary and a ``where=``
predicate all resolve the same way, per axis.
"""

import numpy as np
import pytest

import jno
from jno.trace import mesh_velocity


def _dom(size=0.3, t=(0.0, 0.2, 5)):
    return jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain(time=t)


def test_a_coordinate_time_derivative_is_a_geometry_term():
    """``yb.d(tb) - v`` is recognised structurally — by containing d(spatial)/d(temporal), so the existing
    ``Variable.d`` is all that is needed to express it."""
    d = _dom()
    xb, yb, tb = d.variable("boundary", split=True)
    found = mesh_velocity(yb.d(tb) - 0.5)
    assert found is not None, "a coordinate time-derivative term must classify as geometry"
    coord, tvar, _jac = found
    assert coord.tag == "boundary"
    assert coord.dim[0] == 1, "must identify the AXIS -- y is column 1, and tagging is per-axis"
    assert tvar.axis == "temporal"


def test_geometry_terms_are_not_boundary_specific():
    """The generality that matters: an interior region and a ``where=`` predicate classify identically.
    ``domain.variable`` resolves interior / boundary / predicate the same way, so mesh motion is not a
    boundary feature that happens to be reusable — it is a coordinate feature."""
    d = _dom()
    xi, yi, ti = d.variable("interior", split=True)
    xc, yc, tc = d.variable("core", where=lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.04, split=True)

    interior = mesh_velocity(xi.d(ti) - 0.3 * (yi - 0.5))
    core = mesh_velocity(yc.d(tc) - 1.0)
    assert interior is not None and interior[0].tag == "interior"
    assert interior[0].dim[0] == 0, "xi moves the x column"
    assert core is not None and core[0].tag == "core"
    assert core[0].dim[0] == 1, "yc moves the y column"


def test_a_velocity_may_read_the_solved_field():
    """The point of putting the law in the term list: it is ordinary traced math, so an interface law can
    reference the solution (a Stefan front ``v_n = -(k/L)·∇T·n``) instead of a Python callback."""
    d = _dom()
    u, _v = d.fem_symbols()
    xb, yb, tb, nx, ny = d.variable("boundary", normals=True, split=True)  # (x, y, t, nx, ny)
    tf = u.bind(x=xb, y=yb).freeze(np.zeros(len(d.mesh.points)))
    term = yb.d(tb) - (-(0.5) * (tf.x * nx + tf.y * ny)) * ny
    found = mesh_velocity(term)
    assert found is not None, "a state-dependent velocity is still a geometry term"
    assert found[0].dim[0] == 1


def test_ordinary_terms_are_never_claimed_as_geometry():
    """The classifier must not steal constraints. A weak form is poisoned by its test function even though
    its integrand mentions a coordinate derivative; a Dirichlet BC and a plain expression carry no
    coordinate time-derivative at all."""
    d = _dom()
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)

    for label, term in [
        ("weak form", ui.t * vi + ui.x * vi.x + ui.y * vi.y),
        ("Dirichlet", u(xb, yb) - 0.0),
        ("initial condition", u(ci[0], ci[1]) - 1.0),
        ("plain coordinate expression", yb - 0.5),
    ]:
        assert mesh_velocity(term) is None, f"{label} must not classify as a geometry term"


def test_one_geometry_term_moves_one_coordinate():
    """``xb.d(tb) + yb.d(tb) - 1`` under-determines the motion — one residual, two unknowns. Fail loud and
    say the fix, rather than silently moving whichever coordinate the walk happened to find first."""
    d = _dom()
    xb, yb, tb = d.variable("boundary", split=True)
    with pytest.raises(ValueError, match="may move ONE coordinate"):
        mesh_velocity(xb.d(tb) + yb.d(tb) - 1.0)


def test_the_rest_of_the_problem_is_unaffected():
    """A geometry term is pulled out BEFORE weak-form / Dirichlet classification, so the FE problem it
    accompanies has exactly the DOFs and mode it would have had on its own."""
    d = _dom()
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    physics = [ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0]

    plain = jno.fem(list(physics))
    moving = jno.fem([*physics, yb.d(tb) - 0.5])

    assert moving._mode == plain._mode == "transient"
    assert moving.dofs == plain.dofs, "a geometry term must not add or consume an FE unknown"
    assert len(moving._geometry) == 1 and len(plain._geometry) == 0


def test_an_unwired_geometry_term_refuses_to_solve():
    """Until the motion driver lands, solving would leave the mesh where it started and return a
    confidently wrong answer. Refuse instead — a boundary that silently does not move has no symptom."""
    d = _dom()
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0, yb.d(tb) - 0.5])
    with pytest.raises(NotImplementedError, match="mesh-motion driver"):
        fem.solve()
