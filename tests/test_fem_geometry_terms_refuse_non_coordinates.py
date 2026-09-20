"""A geometry term moves a mesh COORDINATE by a first-order law, and nothing else classifies as one.

:func:`jno.trace.mesh_velocity` recognises a geometry term structurally, by ``d(spatial Variable)/d(time)``.
Three things matched that shape and were silently mis-handled:

* ``u(xb, yb) - yb.d(tb)`` carries the unknown. It was pulled out as a geometry term, so the Dirichlet
  condition it states vanished from the problem with no error.
* ``nx.d(tb)`` and ``cell_size.d(ti)`` differentiate symbols derived FROM the mesh (the outward normal, the
  element size). They are spatial Variables but not coordinates the mesh can be moved along; the driver
  then asked the domain for the vertices of region ``'n_boundary'`` / ``'cell_size'``.
* ``xi.d(ti).d(ti)`` is a second-order law. The walk found the inner ``xi.d(ti)`` and the march integrated
  it as ``x' = ...``, dropping the outer derivative.

Oracle: each is refused by name at classification, i.e. at ``jno.fem([...])``, before anything is solved.
"""

import pytest

import jno
from jno.trace import mesh_velocity


def _dom():
    return jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain(time=(0.0, 0.2, 5))


def test_a_term_carrying_the_unknown_is_refused_not_taken_as_mesh_motion():
    d = _dom()
    u, _v = d.fem_symbols()
    xb, yb, tb = d.variable("boundary", split=True)
    with pytest.raises(ValueError, match="carries the unknown"):
        mesh_velocity(u(xb, yb) - yb.d(tb))


def test_a_term_carrying_the_unknown_is_refused_at_build():
    """The same refusal through the front door: the Dirichlet condition must not silently disappear."""
    d = _dom()
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    with pytest.raises(ValueError, match="carries the unknown"):
        jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - yb.d(tb), u(x0, y0) - 1.0])


@pytest.mark.parametrize("which", ["normal", "cell_size"])
def test_a_symbol_derived_from_the_mesh_is_not_a_coordinate_to_move(which):
    d = _dom()
    _xi, _yi, ti = d.variable("interior", split=True)
    _xb, _yb, tb, nx, _ny = d.variable("boundary", normals=True, split=True)
    term = nx.d(tb) - 1.0 if which == "normal" else d.cell_size.d(ti) - 1.0
    with pytest.raises(ValueError, match="not a mesh coordinate"):
        mesh_velocity(term)


def test_a_second_order_law_is_refused():
    d = _dom()
    xi, _yi, ti = d.variable("interior", split=True)
    with pytest.raises(ValueError, match="second time derivative"):
        mesh_velocity(xi.d(ti).d(ti) - 1.0)


def test_a_first_order_law_still_classifies():
    """The refusals must not catch the terms the driver does integrate."""
    d = _dom()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb, nx, ny = d.variable("boundary", normals=True, split=True)
    for term, tag in [
        (xi.d(ti) - 0.3 * (yi - 0.5), "interior"),
        (yb.d(tb) - 0.05 * ny, "boundary"),
        (xb.d(tb) - 0.1 * nx * d.cell_size, "boundary"),
    ]:
        coord, _t, _jac = mesh_velocity(term)
        assert coord.tag == tag
