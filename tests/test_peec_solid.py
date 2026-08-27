"""Solid conductors: a box becomes a lattice of bars, and bars weld to wires where the metal touches.

A solid has no centreline, so a line's discretisation does not apply. The volume is cut into a
regular grid; the nodes are cell centres and the elements are the bars joining adjacent centres.
The oracle for conduction is exact: with nodes at cell centres the conducting span is
``extent - pitch``, and the DC resistance over that span is ``span / (sigma * W * T)``.
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.peec import bar_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

SIG = 5.8e7
LX, WY, TZ = 0.040, 0.004, 0.002


def ends(f):
    p = np.asarray(f.nodes)
    return (
        terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9),
        terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9),
    )


@pytest.mark.parametrize("pitch", [0.004, 0.002, 0.001])
def test_the_lattice_conducts_exactly(pitch):
    f = bar_filaments(jno.Shape.box(0, 0, 0, LX, WY, TZ), size=pitch)
    a, b = ends(f)
    _cur, _phi, inj = solve_network(f, SIG, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=0.0)
    p = np.asarray(f.nodes)[:, 0]
    span = p.max() - p.min()
    assert abs((1.0 / complex(inj["A"]).real) / (span / (SIG * WY * TZ)) - 1) < 1e-10


def test_the_lattice_is_a_grid_with_one_bar_family_per_axis():
    f = bar_filaments(jno.Shape.box(0, 0, 0, LX, WY, TZ), size=0.002)
    n = f.lattice["n"]
    assert n == (20, 2, 1)
    counts = np.bincount(f.lattice["axis"], minlength=3)
    # (nx-1)*ny*nz along x, nx*(ny-1)*nz along y, and none along z where the box is one cell thick
    assert counts.tolist() == [(n[0] - 1) * n[1] * n[2], n[0] * (n[1] - 1) * n[2], 0]
    assert np.allclose(np.asarray(f.nodes).shape[0], n[0] * n[1] * n[2])


def test_a_pitch_that_leaves_one_cell_everywhere_is_refused():
    with pytest.raises(ValueError, match="no bar joins two cells"):
        bar_filaments(jno.Shape.box(0, 0, 0, 0.01, 0.01, 0.01), size=0.02)


def test_a_solid_solves_through_the_front_door():
    bar = jno.Shape.box(0, 0, 0, LX, WY, TZ, size=0.001).attach(sigma=SIG).name("bar")
    d = bar.domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: x > LX - 0.0011)
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=0.0).solve()
    assert abs(float(sol.R) / ((LX - 0.001) / (SIG * WY * TZ)) - 1) < 1e-10
    assert float(sol.L) > 0


def test_a_wire_landing_on_a_trace_carries_current():
    """The mixed case: separate discretisations, welded where the geometry says the metal touches."""
    trace = jno.Shape.box(0, 0, 0, 0.02, 0.004, 0.001, size=0.001).attach(sigma=SIG).name("trace")
    wire = (
        jno.Shape.line([(0.019, 0.002, 0.0005), (0.019, 0.002, 0.006), (0.030, 0.002, 0.0005)], r=1.9e-4, size=0.001)
        .attach(sigma=SIG)
        .name("wire")
    )
    d = (trace + wire).domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: (x > 0.0295) & (z < 0.0011))
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    sol = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=0.0).solve()
    assert np.isfinite(float(sol.R)) and float(sol.R) > 0
    assert abs(complex(sol.current("A"))) > 1.0  # current actually crosses the joint


def test_conductors_that_do_not_touch_are_refused_rather_than_returning_infinity():
    """A singular solve returns inf without complaining, and an infinite resistance reads physical."""
    a = jno.Shape.box(0, 0, 0, 0.01, 0.004, 0.001, size=0.001).attach(sigma=SIG).name("a")
    b = jno.Shape.box(0.02, 0, 0, 0.03, 0.004, 0.001, size=0.001).attach(sigma=SIG).name("b")
    d = (a + b).domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: x > 0.0289)
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    with pytest.raises(ValueError, match="no conducting path between terminals"):
        jno.peec([v(*at("A")) - v(*at("B")) - 1.0]).solve()


def test_an_unsupported_conductor_shape_says_which_two_exist():
    sph = jno.Shape.sphere(0, 0, 0, 0.01, size=0.002).attach(sigma=SIG).name("blob")
    box = jno.Shape.box(0.02, 0, 0, 0.03, 0.004, 0.001, size=0.001).attach(sigma=SIG).name("b")
    d = (sph + box).domain()
    d.tag("A", lambda x, y, z: x < -0.005)
    d.tag("B", lambda x, y, z: x > 0.0289)
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]
    with pytest.raises(NotImplementedError, match="Shape.line becomes filaments"):
        jno.peec([v(*at("A")) - v(*at("B")) - 1.0]).solve()
