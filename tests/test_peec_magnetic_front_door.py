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
