"""What a region CARRIES decides what it is -- there is no mode flag and no solver argument.

`Shape.attach(**props)` is already generic, so a magnetic material needs no new mechanism:

    sigma        a conductor
    mu_r         a core that does not conduct -- ferrite, a powder core
    both         a conducting magnetic material -- a lamination, a lossy core
    neither      not solvable, and said so

The coupled magnetic solve is not wired yet. Until it is, a network carrying `mu_r` is REFUSED at
solve time rather than answered: returning the coreless number with no sign the core was dropped is
precisely the silent-wrong-answer this solver exists not to give.
"""

import jax
import numpy as np
import pytest

import jno

jax.config.update("jax_enable_x64", True)

CU = 5.8e7


def _net(core=None, both=False, wire=False):
    bar = jno.Shape.box(0, 0, 0, 0.020, 0.004, 0.002, size=(0.002, 0.002, 0.002)).attach(sigma=CU).name("bar")
    sh = bar
    if core is not None:
        props = {"mu_r": core} | ({"sigma": CU} if both else {})
        sh = sh + jno.Shape.box(0, 0.008, 0, 0.020, 0.014, 0.002, size=(0.002, 0.002, 0.002)).attach(
            **props
        ).name("core")
    if wire:
        sh = sh + jno.Shape.line(
            [(0.002, 0.002, 0.002), (0.002, 0.002, 0.005), (0.018, 0.002, 0.002)], r=2e-4, size=0.002
        ).attach(sigma=CU).name("w")
    d = sh.domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: x > 0.0189)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    return jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e5)


def test_a_core_is_discretised_into_its_own_mesh():
    """A core is voxels and the faces between them, exactly as a conductor is -- the same builder."""
    e = _net(core=2000.0).build()
    assert e.mag is not None and e.mag_names == ("core",)
    assert int(np.asarray(e.mag.length).size) > 0
    assert int(np.asarray(e.fil.length).size) > 0  # and the conductor is still its own mesh


def test_a_network_with_no_core_is_untouched():
    """The governing constraint: with no mu_r anywhere, nothing about the old path changes."""
    e = _net().build()
    assert e.mag is None and e.mag_names == ()
    assert float(np.real(e.solve().R)) > 0


def test_a_conducting_magnetic_region_is_in_BOTH_meshes():
    """A lamination conducts and carries flux; it is not a choice between the two."""
    plain = _net(core=2000.0).build()
    both = _net(core=2000.0, both=True).build()
    assert int(np.asarray(both.mag.length).size) == int(np.asarray(plain.mag.length).size)
    assert int(np.asarray(both.fil.length).size) > int(np.asarray(plain.fil.length).size)


def test_a_region_carrying_neither_property_is_refused():
    """Silence here would mean a region quietly excluded from a solve it was drawn into."""
    bar = jno.Shape.box(0, 0, 0, 0.020, 0.004, 0.002, size=(0.002,) * 3).attach(sigma=CU).name("bar")
    lost = jno.Shape.box(0, 0.008, 0, 0.020, 0.014, 0.002, size=(0.002,) * 3).name("lost")
    d = (bar + lost).domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: x > 0.0189)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    with pytest.raises(ValueError, match="neither a conductivity nor a permeability"):
        jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e5).build()


def test_a_core_on_a_filament_is_refused():
    """A filament has no cross-section, so there is nothing for flux to pass through."""
    sh = jno.Shape.line([(0, 0, 0), (0.02, 0, 0)], r=2e-4, size=0.002).attach(mu_r=2000.0).name("bad")
    d = sh.domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: x > 0.0189)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    with pytest.raises(NotImplementedError, match="carries FLUX through a cross-section"):
        jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e5).build()


def test_solving_with_a_core_refuses_rather_than_dropping_it():
    """The whole point of landing the front door before the physics.

    A coreless answer returned for a model with a core would be wrong with nothing to show for it.
    """
    e = _net(core=2000.0).build()
    with pytest.raises(NotImplementedError, match="COUPLED"):
        e.solve()


def test_a_core_in_a_welded_network_is_refused_for_now():
    """Welding already needs a cross block and a whole-system factorisation; a coupled magnetic
    system on top of that is untested, so it is refused rather than guessed at."""
    with pytest.raises(NotImplementedError, match="together with a Shape.line"):
        _net(core=2000.0, wire=True).build()


def test_a_unit_permeability_region_is_air_and_says_so():
    """chi = mu_r - 1 is zero, so the region adds no magnetisation and is not a core.

    Dropping it is exact rather than an approximation. With nothing else attached it is air with a
    name, which is a modelling slip worth refusing rather than solving around.
    """
    bar = jno.Shape.box(0, 0, 0, 0.020, 0.004, 0.002, size=(0.002,) * 3).attach(sigma=CU).name("bar")
    air = jno.Shape.box(0, 0.008, 0, 0.020, 0.014, 0.002, size=(0.002,) * 3).attach(mu_r=1.0).name("air")
    d = (bar + air).domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: x > 0.0189)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    with pytest.raises(ValueError, match="air with a name"):
        jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e5).build()


def test_a_unit_permeability_conductor_still_conducts():
    """The same region carrying a sigma is a perfectly good conductor; only the core part is air."""
    bar = jno.Shape.box(0, 0, 0, 0.020, 0.004, 0.002, size=(0.002,) * 3).attach(sigma=CU).name("bar")
    both = jno.Shape.box(0, 0.008, 0, 0.020, 0.014, 0.002, size=(0.002,) * 3).attach(
        mu_r=1.0, sigma=CU
    ).name("both")
    d = (bar + both).domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: x > 0.0189)
    i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(2, None))[:3]
    e = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e5).build()
    assert e.mag is None  # no magnetisation to carry, so no magnetic mesh and no refusal
    assert float(np.real(e.solve().R)) > 0


def test_the_two_meshes_land_on_one_grid():
    """The conductors and the core must share a lattice, or the coupling between them is not
    Toeplitz and the FFT that applies it does not exist.

    Built independently they do NOT share one: each call takes its extent and its pitch from its own
    regions, so the grids come out different sizes and offset from each other -- measured (10,2,1)
    against (10,3,1) on this very geometry. Each mesh is now framed by the other's shapes, which
    fixes the grid without putting the other's cells into its occupancy.
    """
    e = _net(core=2000.0).build()
    le, lm = e.fil.lattice, e.mag.lattice
    assert le["n"] == lm["n"], (le["n"], lm["n"])
    assert le["d"] == lm["d"], (le["d"], lm["d"])

    # and the occupancies stay disjoint: framing is not meshing
    ec = np.asarray(le["cells"])
    mc = np.asarray(lm["cells"])
    assert ec.shape == mc.shape
    assert int((ec & mc).sum()) == 0
    assert int(ec.sum()) > 0 and int(mc.sum()) > 0


def test_a_conductor_only_network_keeps_the_grid_it_always_had():
    """Framing must be inert when there is nothing to frame -- this is the regression guard for
    every existing model, which must see byte-identical geometry."""
    from jno.utils.solver.peec import bar_filaments

    sh = jno.Shape.box(0, 0, 0, 0.020, 0.004, 0.002, size=(0.002,) * 3).attach(sigma=CU).name("bar")
    a = bar_filaments(sh, sigma=CU)
    b = bar_filaments(sh, sigma=CU, grid_shapes=())
    assert a.lattice["n"] == b.lattice["n"] and a.lattice["d"] == b.lattice["d"]
    assert np.allclose(np.asarray(a.nodes), np.asarray(b.nodes))
