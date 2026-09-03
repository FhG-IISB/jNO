"""A conductor that gets no cell is a different circuit, not a coarser one.

A lattice takes a cell when its CENTRE is inside the geometry. A solid thinner than the pitch can
therefore fall between two rows of centres and contribute nothing at all -- while every other
conductor meshes normally, the network solves, and the answer describes a circuit missing a part.

Found on a real power module: 0.57 mm traces force a one-cell-thick z pitch (which the
surface-impedance guard REQUIRES at MHz), and the 0.18 mm dies then land between cell centres at
3.34 / 3.86 / 4.39 mm. All four vanished. The symptom was that changing the die conductivity by a
factor of 38,000 -- a MOSFET channel against solid copper -- moved the loop inductance and the
resistance by not one bit, because there was nothing there to change.
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.peec import bar_filaments

jax.config.update("jax_enable_x64", True)

CU = 5.8e7


def test_a_solid_thinner_than_the_pitch_is_refused():
    """The power-module case, reduced: a thin die on a thick trace, one cell through the trace."""
    trace = jno.Shape.box(0, 0, 0.0, 0.020, 0.010, 0.00057).attach(sigma=CU).name("trace")
    die = jno.Shape.box(0.006, 0.003, 0.00057, 0.010, 0.007, 0.00075).attach(sigma=CU).name("die")
    with pytest.raises(ValueError, match="got no cell of this lattice"):
        bar_filaments([trace, die], size=(0.001, 0.001, 0.00057), sigma=[CU, CU])


def test_the_refusal_names_the_conductor_and_how_thin_it_is():
    """A guard that says only "something is wrong" costs as much time as no guard."""
    trace = jno.Shape.box(0, 0, 0.0, 0.020, 0.010, 0.00057).attach(sigma=CU).name("trace")
    die = jno.Shape.box(0.006, 0.003, 0.00057, 0.010, 0.007, 0.00075).attach(sigma=CU).name("die")
    with pytest.raises(ValueError) as e:
        bar_filaments([trace, die], size=(0.001, 0.001, 0.00057), sigma=[CU, CU])
    msg = str(e.value)
    assert "'die'" in msg, msg  # by NAME, not by index
    assert "0.32 cells across" in msg or "0.32)" in msg, msg  # and by how badly it missed


def test_a_solid_drawn_inside_another_is_refused_too():
    """The other cause: cells go to the FIRST solid containing them, so a piece drawn inside another
    never gets any -- and its own conductivity is then silently never applied."""
    outer = jno.Shape.box(0, 0, 0, 0.020, 0.010, 0.004).attach(sigma=CU).name("outer")
    inner = jno.Shape.box(0.005, 0.002, 0.001, 0.010, 0.006, 0.003).attach(sigma=CU).name("inner")
    with pytest.raises(ValueError, match="got no cell of this lattice"):
        bar_filaments([outer, inner], size=(0.001,) * 3, sigma=[CU, CU / 100])


def test_a_finer_pitch_meshes_it_and_is_not_refused():
    """The fix the message recommends has to actually work, or the guard is just an obstacle."""
    trace = jno.Shape.box(0, 0, 0.0, 0.020, 0.010, 0.00057).attach(sigma=CU).name("trace")
    die = jno.Shape.box(0.006, 0.003, 0.00057, 0.010, 0.007, 0.00075).attach(sigma=CU).name("die")
    f = bar_filaments([trace, die], size=(0.001, 0.001, 0.00009), sigma=[CU, CU])
    part = np.asarray(f.part)
    assert int((part == 0).sum()) > 0 and int((part == 1).sum()) > 0


def test_an_ordinary_layout_is_untouched():
    """The regression guard: every model that already meshed must still mesh."""
    a = jno.Shape.box(0, 0, 0, 0.020, 0.004, 0.002).attach(sigma=CU).name("a")
    b = jno.Shape.box(0, 0.006, 0, 0.020, 0.010, 0.002).attach(sigma=CU).name("b")
    f = bar_filaments([a, b], size=(0.002,) * 3, sigma=[CU, CU])
    assert set(np.unique(np.asarray(f.part)).tolist()) == {0, 1}
