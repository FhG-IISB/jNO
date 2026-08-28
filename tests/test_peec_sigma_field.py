"""Conductivity as a FIELD: ``.attach(sigma=...)`` takes a function of position or a per-cell vector.

A conductivity is a design variable at least as often as it is a material constant, and the useful
design variable varies *within* a conductor -- that is what a density (SIMP) topology optimisation
optimises, and it is the one thing a per-conductor scalar cannot express. The rule that makes this
land without a special case already existed: a bar joins two cells, its halves are in series, so its
conductivity is their harmonic mean. Fixing the conductivity to the CELLS rather than to the
conductor generalises it exactly, and one conductivity per conductor stays a special case of it.

The oracle here is conduction itself, not a restatement of that rule: on a one-cell-wide chain every
bar is in series, so the DC resistance is the sum over bars of ``(d/2)/(sigma_a A) + (d/2)/(sigma_b
A)`` -- half the bar at each end cell's conductivity. That is the physics the harmonic mean encodes,
written independently of it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.peec import bar_filaments, element_centres, line_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

SIG = 5.8e7
LX, WY, TZ = 0.040, 0.002, 0.002
PITCH = 0.002  # a one-cell-wide chain: 20 cells along x, one across y and z


def chain(sigma):
    """The 1-D bar chain, with whatever conductivity spelling is under test."""
    return bar_filaments(jno.Shape.box(0, 0, 0, LX, WY, TZ), size=PITCH, sigma=sigma)


def dc_resistance(f):
    a, b = (
        terminal_nodes(f, lambda q: q[:, 0] < np.asarray(f.nodes)[:, 0].min() + 1e-9),
        terminal_nodes(f, lambda q: q[:, 0] > np.asarray(f.nodes)[:, 0].max() - 1e-9),
    )
    _c, _p, inj = solve_network(f, f.lattice["sigma"], {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=0.0)
    return 1.0 / complex(inj["A"]).real


def series_oracle(f, fn):
    """Sum the two half-bars in series, straight from the geometry. Independent of the code path."""
    inc = np.asarray(f.incidence.toarray())
    ia, ib = np.argmax(inc == 1, axis=0), np.argmax(inc == -1, axis=0)
    x = np.asarray(f.nodes)[:, 0]
    ln, ar = np.asarray(f.length), np.asarray(f.area)
    return float((0.5 * ln / (fn(x[ia]) * ar) + 0.5 * ln / (fn(x[ib]) * ar)).sum())


# --- a field reduces to the scalar it generalises ------------------------------------------------


def test_a_constant_field_is_the_scalar_exactly():
    """The generalisation has to be one: a flat field must not perturb the old answer at all."""
    ref = np.asarray(chain(SIG).lattice["sigma"])
    for spelling in (lambda x, y, z: SIG * jnp.ones_like(x), lambda x, y: SIG + 0 * x, lambda x: SIG + 0 * x):
        assert np.array_equal(np.asarray(chain(spelling).lattice["sigma"]), ref)


def test_a_constant_vector_is_the_scalar_exactly():
    n = len(chain(SIG).nodes)
    got = chain(jnp.full(n, float(SIG))).lattice["sigma"]
    assert np.array_equal(np.asarray(got), np.asarray(chain(SIG).lattice["sigma"]))


def test_a_flat_field_conducts_like_the_uniform_bar():
    """Same closed form the uniform lattice is pinned to: span / (sigma W T)."""
    f = chain(lambda x, y, z: SIG * jnp.ones_like(x))
    p = np.asarray(f.nodes)[:, 0]
    assert abs(dc_resistance(f) / ((p.max() - p.min()) / (SIG * WY * TZ)) - 1) < 1e-10


# --- a field that actually varies -----------------------------------------------------------------


def test_a_graded_chain_is_the_series_sum():
    """A smoothly graded conductivity, against half-bars summed in series."""
    fn = lambda x: SIG * (0.2 + 0.8 * x / LX)  # noqa: E731
    f = chain(lambda x, y, z: SIG * (0.2 + 0.8 * x / LX))
    assert abs(dc_resistance(f) / series_oracle(f, fn) - 1) < 1e-10


def test_two_materials_in_series_are_two_resistors_in_series():
    """The extreme of a graded field: a step. Closed form, no summation at all.

    Nodes sit at cell centres, so the conducting span stops half a pitch inside each end. With the
    step on a cell boundary each half-span is one material, and the straddling bar is the two
    quarter-spans that join them.
    """
    lo, hi = SIG, SIG / 4
    f = chain(lambda x, y, z: jnp.where(x < 0.5 * LX, lo, hi))
    span = LX - PITCH
    exact = 0.5 * span / (lo * WY * TZ) + 0.5 * span / (hi * WY * TZ)
    assert abs(dc_resistance(f) / exact - 1) < 1e-10
    # and it must sit strictly between the two uniform bars it is made of
    assert dc_resistance(chain(lo)) < dc_resistance(f) < dc_resistance(chain(hi))


def test_a_void_cell_is_nearly_an_open_circuit():
    """The SIMP extreme: density -> 0 on one cell of the chain.

    Nodes are cell CENTRES, so the void is picked at one -- 0.021, half a pitch off the midpoint --
    rather than at 0.5*LX, which is a cell boundary and would select nothing.
    """
    mid = 0.021
    r = [dc_resistance(chain(lambda x, y, z: SIG * jnp.where(abs(x - mid) < 0.5 * PITCH, e, 1.0))) for e in (1e-3, 1e-6)]
    assert r[1] / r[0] > 500  # a thousandth of the density is a thousand times the resistance there
    assert np.isfinite(r[1])  # and it stays a number: a void is a bad conductor, not a singular one


def test_a_field_on_one_conductor_leaves_its_neighbour_alone():
    """Two conductors share one grid; a field is resolved over its OWN cells, not the whole lattice."""
    left = jno.Shape.box(0, 0, 0, 0.02, WY, TZ).name("left")
    right = jno.Shape.box(0.02, 0, 0, 0.04, WY, TZ).name("right")
    f = bar_filaments([left, right], size=PITCH, sigma=[lambda x, y, z: SIG * jnp.ones_like(x), SIG])
    assert np.allclose(np.asarray(f.lattice["sigma"]), SIG)
    graded = bar_filaments([left, right], size=PITCH, sigma=[lambda x, y, z: SIG * (0.1 + 0 * x), SIG])
    s = np.asarray(graded.lattice["sigma"])
    assert s.min() == pytest.approx(0.1 * SIG) and s.max() == pytest.approx(SIG)


# --- lines take a field too -----------------------------------------------------------------------


def test_element_centres_are_the_midpoints():
    f = line_filaments(jno.Shape.line([(0, 0, 0), (0, 0, 0.05)], r=5e-4, size=0.005))
    c = np.asarray(element_centres(f))
    inc = np.asarray(f.incidence.toarray())
    ia, ib = np.argmax(inc == 1, axis=0), np.argmax(inc == -1, axis=0)
    nodes = np.asarray(f.nodes)
    assert np.allclose(c, 0.5 * (nodes[ia] + nodes[ib]), atol=1e-14)


def test_a_graded_wire_is_the_series_sum():
    ell = 0.05
    fld = lambda x, y, z: SIG * (0.3 + z / ell)  # noqa: E731  -- POSITIONAL: the third arg is z
    wire = jno.Shape.line([(0, 0, 0), (0, 0, ell)], r=5e-4, size=ell / 10).attach(sigma=fld)
    pads = jno.Shape.sphere(0, 0, 0.0, 1e-3).name("A") + jno.Shape.sphere(0, 0, ell, 1e-3).name("B")
    d = (wire.name("wire") + pads).domain()
    _i, v = d.peec_symbols()
    at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]  # noqa: E731
    got = float(jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=0.0).solve().R)
    f = line_filaments(jno.Shape.line([(0, 0, 0), (0, 0, ell)], r=5e-4, size=ell / 10))
    z = np.asarray(element_centres(f))[:, 2]
    # a wire is one series path: every filament's own resistance, added up
    exact = float((np.asarray(f.length) / ((SIG * (0.3 + z / ell)) * np.asarray(f.area))).sum())
    assert abs(got / exact - 1) < 1e-9


def test_a_short_field_takes_the_FIRST_coordinates():
    """Arity is positional, exactly as it is for an attached FEM coefficient -- ``lambda r, z``
    there means the domain's first two coordinates, not the two named. So a one-argument field is a
    function of x, whatever its parameter is called, and a planar field is ``lambda x, y``.

    Worth pinning rather than assuming: a wire along z given ``lambda z: ...`` is a function of x,
    which is constant along that wire, and the answer would come back plausible and wrong.
    """
    graded = lambda x, y, z: SIG * (0.2 + 0.8 * x / LX)  # noqa: E731
    for short in (lambda x: SIG * (0.2 + 0.8 * x / LX), lambda x, y: SIG * (0.2 + 0.8 * x / LX)):
        assert np.array_equal(np.asarray(chain(short).lattice["sigma"]), np.asarray(chain(graded).lattice["sigma"]))
    # and a field of y alone is flat on this chain, which is one cell wide in y
    assert np.allclose(np.asarray(chain(lambda x, y: SIG * (1 + y)).lattice["sigma"]), SIG * (1 + 0.5 * WY))


def test_the_front_door_takes_a_flat_field_unchanged():
    ell = 0.05
    line = lambda: jno.Shape.line([(0, 0, 0), (0, 0, ell)], r=5e-4, size=ell / 10)  # noqa: E731
    pads = jno.Shape.sphere(0, 0, 0.0, 1e-3).name("A") + jno.Shape.sphere(0, 0, ell, 1e-3).name("B")

    def solve(sig):
        d = (line().attach(sigma=sig).name("wire") + pads).domain()
        _i, v = d.peec_symbols()
        at = lambda t: d.variable(t, split=True, sample=(4, None))[:3]  # noqa: E731
        return complex(jno.peec([v(*at("A")) - v(*at("B")) - 1.0], freq=1e6).solve().Z)

    assert abs(solve(lambda x, y, z: SIG * jnp.ones_like(z)) - solve(SIG)) / abs(solve(SIG)) < 1e-12


# --- it is differentiable, which is the whole reason it exists ------------------------------------


def test_a_density_field_is_differentiable_to_finite_differences():
    """The gradient w.r.t. a design parameter of the field, against a central difference."""

    def resistance(t):
        f = chain(lambda x, y, z: SIG * (0.2 + t * x / LX))
        s = jnp.asarray(f.lattice["sigma"])
        return jnp.sum(jnp.asarray(f.length) / (s * jnp.asarray(f.area)))

    g = float(jax.grad(resistance)(0.8))
    h = 1e-6
    fd = float((resistance(0.8 + h) - resistance(0.8 - h)) / (2 * h))
    assert abs(g / fd - 1) < 1e-7
    assert g < 0  # more density is less resistance


def test_the_solved_port_resistance_is_differentiable():
    """Through the SOLVE, not just the assembly: the adjoint has to carry the field too."""
    a, b = (
        terminal_nodes(chain(SIG), lambda q: q[:, 0] < np.asarray(chain(SIG).nodes)[:, 0].min() + 1e-9),
        terminal_nodes(chain(SIG), lambda q: q[:, 0] > np.asarray(chain(SIG).nodes)[:, 0].max() - 1e-9),
    )

    def port(t):
        f = chain(lambda x, y, z: SIG * (0.2 + t * x / LX))
        _c, _p, inj = solve_network(f, f.lattice["sigma"], {"A": a, "B": b}, [("A", "B", 1.0 + 0j)], omega=0.0)
        return jnp.real(1.0 / inj["A"])

    g = float(jax.grad(port)(0.8))
    h = 1e-6
    assert abs(g / float((port(0.8 + h) - port(0.8 - h)) / (2 * h)) - 1) < 1e-6
    assert g < 0


# --- and it fails loudly -------------------------------------------------------------------------


def test_a_wrong_length_vector_says_how_many_were_needed():
    n = len(chain(SIG).nodes)
    with pytest.raises(ValueError, match=rf"7 conductivities .*discretises into {n} elements"):
        chain(jnp.full(7, float(SIG)))


def test_a_field_that_does_not_broadcast_is_refused():
    with pytest.raises(ValueError, match="one value per element"):
        chain(lambda x, y, z: jnp.ones(3))


def test_the_wrong_number_of_conductors_is_still_refused():
    box = jno.Shape.box(0, 0, 0, LX, WY, TZ)
    with pytest.raises(ValueError, match="2 conductivities for 1 conductors"):
        bar_filaments(box, size=PITCH, sigma=[SIG, SIG])
