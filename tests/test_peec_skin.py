"""The element impedance is a shape-aware SURFACE one, not rho*l/A.

That is what lets a conductor be ONE element through its thickness and still show the skin effect,
which is how the PEEC literature keeps a package tractable (Romano, Kovacevic-Badstuebner, Antonini
& Grossner, IEEE Trans. Electromagn. Compat. 65(2), 2023, sec. II-A). It also strengthens the
preconditioner, whose diagonal is Z_s + j w Lp_aa.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import internal_impedance
from jno.utils.solver.peec import bar_filaments, line_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

SIG, MU0 = 5.8e7, 4e-7 * np.pi


def test_it_is_the_dc_resistance_below_the_skin_depth():
    """A conductor thin against the skin depth conducts through its whole section, as it must."""
    a, ell = 1e-3, 1.0
    dc = ell / (SIG * np.pi * a**2)
    for hz in (0.0, 1.0, 100.0):
        z = complex(internal_impedance(ell, np.pi * a**2, a, True, 2 * np.pi * hz, SIG))
        assert abs(z.real / dc - 1) < 2e-3


def test_a_round_wire_approaches_its_thin_skin_asymptote():
    """Far above the skin depth the current runs in a shell of depth delta: R -> rho / (2 pi a delta)."""
    a, ell = 1e-3, 1.0
    hz = 1e7
    delta = np.sqrt(1.0 / (np.pi * hz * MU0 * SIG))
    asympt = ell / (SIG * 2 * np.pi * a * delta)
    z = complex(internal_impedance(ell, np.pi * a**2, a, True, 2 * np.pi * hz, SIG))
    assert abs(z.real / asympt - 1) < 0.05  # the asymptote drops the curvature correction
    assert z.imag > 0  # and it carries an internal inductance


def test_one_cell_through_the_thickness_still_shows_the_skin_effect():
    """The point of the surface impedance: no splitting across the section, and sqrt(f) still appears."""
    f = bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.004, 0.002), size=(0.002, 0.004, 0.002))
    assert f.lattice["n"][2] == 1  # ONE cell through the 2 mm thickness
    p = np.asarray(f.nodes)
    a = terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9)
    b = terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9)

    rs = []
    for hz in (0.0, 1e5, 1e6, 1e7):
        _c, _phi, inj = solve_network(
            f, SIG, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * hz, matrix_free=False
        )
        rs.append(complex(1.0 / inj["A"]).real)

    assert rs[0] > 0 and all(x > y for x, y in zip(rs[1:], rs[:-1]))  # rises with frequency
    assert rs[3] / rs[0] > 20  # and by a lot: measured 47.9x at 10 MHz
    # deep in the skin regime R grows as sqrt(f), so a decade multiplies it by about sqrt(10)
    assert 2.7 < rs[3] / rs[2] < 3.6


def test_a_wire_and_a_bar_take_different_coefficients():
    """A round section is not a slab: the cylindrical form is 0.02 % where a flat one is 12.5 % out."""
    area, ell, hz = 1e-6, 0.01, 1e7
    z_round = complex(internal_impedance(ell, area, np.sqrt(area / np.pi), True, 2 * np.pi * hz, SIG))
    z_slab = complex(internal_impedance(ell, area, np.sqrt(area), False, 2 * np.pi * hz, SIG))
    assert abs(z_round.real / z_slab.real - 1) > 0.05  # genuinely different, not a shared formula


def test_the_impedance_is_differentiable_through_the_surface_form():
    """The sqrt in gamma is zero at DC, and differentiating a masked branch through it gives NaN."""
    f = line_filaments(jno.Shape.line([(0, 0, 0), (0, 0, 0.03)], r=3e-4, size=0.005))
    for hz in (0.0, 1e6):
        loss = lambda s, hz=hz: jax.numpy.real(
            jax.numpy.sum(internal_impedance(f.length, f.area, f.skin, f.round_, 2 * np.pi * hz, s))
        )
        g = float(jax.grad(loss)(SIG))
        fd = float((loss(SIG * 1.001) - loss(SIG * 0.999)) / (0.002 * SIG))
        assert np.isfinite(g)
        assert abs(g / fd - 1) < 1e-5
        assert g < 0  # more conductive, less resistive


# --------------------------------------------------------------------------------------------
# The condition the surface impedance depends on: the element must BE the whole thickness.
# --------------------------------------------------------------------------------------------


def _bar(n, hz_pitch=None):
    """A 40 x 4 x 2 mm bar cut into ``n`` cells through its 2 mm thickness."""
    t = 0.002
    return bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.004, t), size=(0.002, 0.004, hz_pitch or t / n))


def _port(f, hz):
    p = np.asarray(f.nodes)
    a = terminal_nodes(f, lambda q: q[:, 0] < p[:, 0].min() + 1e-9)
    b = terminal_nodes(f, lambda q: q[:, 0] > p[:, 0].max() - 1e-9)
    _c, _phi, inj = solve_network(f, SIG, {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=2 * np.pi * hz, matrix_free=False)
    return complex(1.0 / inj["A"]).real


def test_the_thickness_is_measured_by_extent_not_by_pitch():
    """A 0.57 mm trace on a 0.5 mm in-plane grid is thin in z and wide in y.

    Picking the thinner PITCH would call the 0.5 mm cell width the thickness and hand the skin
    formula the wrong dimension entirely.
    """
    f = bar_filaments(jno.Shape.box(0, 0, 0, 0.040, 0.020, 0.00057), size=(0.0005, 0.0005, 0.00057))
    assert np.allclose(np.asarray(f.skin), 0.00057)  # the thickness, not the 0.5 mm pitch
    assert np.all(np.asarray(f.span) == 1)


def test_one_cell_spans_the_conductor_and_more_than_one_does_not():
    assert np.all(np.asarray(_bar(1).span) == 1)
    sp = np.asarray(_bar(2).span)
    ax = np.asarray(_bar(2).lattice["axis"])
    assert np.all(sp[ax != 2] == 2)  # in-plane bars: the conductor is two elements thick
    assert np.all(sp[ax == 2] == 1)  # the z bars are thin across y instead, and span that


def test_a_subdivided_element_takes_the_dc_resistance():
    """The free-surface forms do not hold once the interface between two cells is not a surface."""
    ell, area = 0.01, 1e-6
    w = 2 * np.pi * 1e7
    dc = ell / (SIG * area)
    z1 = complex(internal_impedance(ell, area, np.sqrt(area), False, w, SIG, span=1))
    z2 = complex(internal_impedance(ell, area, np.sqrt(area), False, w, SIG, span=2))
    assert z1.real > 3 * dc  # deep in the skin regime, the surface form is far above DC
    assert abs(z2.real / dc - 1) < 1e-12 and z2.imag == 0.0  # subdivided: exactly rho l / A


def test_stacking_elements_no_longer_doubles_the_conductance():
    """The defect this guards. Every element taking the surface form gave EXACTLY half the R.

    Measured before the fix, on this bar at 1 MHz and 10 MHz: 1 cell 1239 / 3919 uOhm, 2 cells
    620 / 1959 -- a factor of 0.500 both times, because each half-thickness cell counted the
    interface it shares with the other as a free surface.
    """
    assert abs(_port(_bar(1), 1e6) / 1239.25e-6 - 1) < 0.01  # the one-cell model is unchanged
    with pytest.raises(ValueError, match="Neither model applies"):
        _port(_bar(2), 1e6)


def test_the_unresolvable_middle_is_refused_not_returned():
    """Subdivided AND too coarse satisfies neither model, so it raises rather than reporting DC."""
    with pytest.raises(ValueError, match="elements thick where each is"):
        _port(_bar(2), 1e6)
    with pytest.raises(ValueError, match="skin depths through it"):
        _port(_bar(4), 1e6)


def test_a_thin_conductor_may_still_be_subdivided():
    """Below a couple of skin depths there is nothing to lose, so a split conductor is fine."""
    assert _port(_bar(2), 1e2) > 0  # 2 mm against a 6.6 mm skin depth: no refusal
    assert _port(_bar(4), 1e2) > 0


def test_dc_never_refuses():
    """At zero frequency every element is rho l / A and the thickness does not enter."""
    for n in (1, 2, 4):
        assert abs(_port(_bar(n), 0.0) / _port(_bar(1), 0.0) - 1) < 0.02


def test_the_two_valid_models_agree_with_each_other():
    """The real check: one element with a surface impedance, against many that resolve the depth.

    Different mechanisms entirely -- a closed form on one element, versus a current distribution the
    solve finds for itself -- so agreeing to a few percent is evidence both are right.
    """
    hz = 1e4  # skin depth 0.661 mm, so 0.0625 mm cells resolve it and 2 mm does not
    surface = _port(_bar(1), hz)
    resolved = _port(_bar(32), hz)
    assert abs(surface / resolved - 1) < 0.05


def test_a_run_breaks_where_the_MATERIAL_changes():
    """A die on a trace is 2 cells of metal but ONE cell of each conductor.

    The surface impedance asks whether an element spans its own conductor, and the skin effect does
    not run across a junction between different materials -- so the run that counts elements through
    a thickness has to break there. It did not, and that refused a perfectly good model: a die
    projected onto a slab above its pad read as a 2-cell-thick conductor and the solve was rejected.

    Stacked pieces of the SAME material stay one run, which matters just as much: a terminal post
    standing on a trace really is one column of copper, and that case is a genuine warning.
    """
    lo = jno.Shape.box(0, 0, 0, 0.020, 0.008, 0.001)
    hi = jno.Shape.box(0, 0, 0.001, 0.020, 0.008, 0.002)
    grid = dict(size=(0.002, 0.002, 0.001))

    same = bar_filaments([lo, hi], sigma=[SIG, SIG], **grid)
    diff = bar_filaments([lo, hi], sigma=[SIG, 5.0e3], **grid)
    ax_s, ax_d = np.asarray(same.lattice["axis"]), np.asarray(diff.lattice["axis"])
    # in-plane bars are the ones whose thin direction is the stack
    assert set(np.asarray(same.span)[ax_s != 2].tolist()) == {2}  # one conductor, two cells thick
    assert set(np.asarray(diff.span)[ax_d != 2].tolist()) == {1}  # two conductors, one cell each


def test_the_material_break_does_not_silence_the_stacked_copper_warning():
    """The post-on-a-trace case must still be caught: same metal, so still one conductor."""
    trace = jno.Shape.box(0, 0, 0, 0.040, 0.010, 0.00057)
    post = jno.Shape.box(0.018, 0.003, 0.00057, 0.022, 0.007, 0.00157)
    f = bar_filaments([trace, post], size=(0.002, 0.002, 0.00057), sigma=[SIG, SIG])
    assert (np.asarray(f.span) > 1).any()  # the column under the post is more than one element


def test_the_thickness_guard_runs_at_build_and_does_not_block_a_jit(caplog):
    """A guard that stops guarding the moment you jit is worse than none, so it moved to `build()`.

    It reads the conductivity, which inside a jit has no value at all. Running it once at build
    against the DECLARED conductivity is the conservative case: the verdict turns on the skin depth,
    a design variable only ever lowers sigma, and a lower sigma is a deeper skin depth and so a
    milder verdict. On the example module this was the last thing keeping a welded network -- a
    lattice plus bond wires -- out of a jit, worth 2.5x on a design iteration.
    """
    import logging

    # The shape it was written for: a thin trace, one cell through, with a THICK POST standing on
    # it -- a terminal post on a 0.57 mm trace is exactly this. A minority is unresolved, so the
    # guard warns rather than refusing the package, which is the case worth keeping alive.
    h = 0.0005
    trace = jno.Shape.box(0, 0, 0, 0.02, 0.004, h, size=(0.001, 0.001, h)).attach(sigma=SIG).name("trace")
    post = jno.Shape.box(0.009, 0.001, h, 0.011, 0.003, h + 0.0015).attach(sigma=SIG).name("post")
    d = (trace + post).domain()
    d.tag("A", lambda x, y, z: x < 0.0011)
    d.tag("B", lambda x, y, z: x > 0.0189)
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]  # noqa: E731

    with caplog.at_level(logging.WARNING):
        built = jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6).build()
    assert "skin depth" in caplog.text, "the guard did not run at build"
    caplog.clear()

    # and the solve it guards now goes through a jit, with the same answer
    f = lambda s: jnp.real(built.solve(sigma={"trace": SIG * s, "post": SIG * s}).Z)  # noqa: E731
    assert float(jax.jit(f)(1.0)) == pytest.approx(float(f(1.0)), rel=1e-12)
    assert "skin depth" not in caplog.text, "the guard repeated itself on every solve"
