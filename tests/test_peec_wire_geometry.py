"""Bond-wire geometry as a design variable: `line_filaments(points=...)` differentiates.

The r-adaptivity contract, applied to a polyline conductor. What is fixed is combinatorial -- how
many filaments a segment is cut into, and which endpoints are the same node -- and it is decided
from the shape's own vertices. What moves is everything geometric, computed in jax from vertices
that may be traced, so a gradient reaches the routing itself.

The lattice cannot do this: its occupancy is a discrete in/out test per cell, and the FFT needs a
regular grid, so its nodes cannot move at all. Wires can, and on a power module they are where the
loop inductance lives.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import pair_matrix
from jno.utils.solver.peec import line_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

SIG, MM, RW = 5.8e7, 1e-3, 1.9e-4
ARC = [(0, 0, 0), (5 * MM, 0, 2 * MM), (10 * MM, 0, 0)]


def _shape():
    return jno.Shape.line(ARC, r=RW, size=1 * MM)


def _loop(points):
    return line_filaments(_shape(), points=None if points is None else [points])


# Terminals are resolved ONCE, from the reference geometry: `terminal_nodes` reads coordinates, and
# under a gradient those are tracers. Which nodes a pad owns is structural, exactly like the node
# numbering -- so it is fixed at the reference configuration and the vertices move under it.
_REF = _loop(None)
TERM = {
    "A": terminal_nodes(_REF, lambda q: np.linalg.norm(q - np.array(ARC[0]), axis=1) < 1e-9),
    "B": terminal_nodes(_REF, lambda q: np.linalg.norm(q - np.array(ARC[-1]), axis=1) < 1e-9),
}


def _inductance(points, hz=1e7):
    f, term = _loop(points), TERM
    _c, _p, inj = solve_network(f, SIG, term, [("A", "B", 1.0 + 0j)], (), (), (), omega=2 * np.pi * hz, matrix_free=False)
    return jnp.imag(1.0 / inj["A"]) / (2 * np.pi * hz)


def test_supplied_points_reproduce_the_shapes_own():
    """Passing the vertices the shape already has must change nothing at all."""
    a = _loop(None)
    b = _loop(jnp.asarray(ARC, dtype=float))
    for k in ("pos", "mom", "length", "area", "nodes", "self_g", "skin"):
        assert np.allclose(np.asarray(getattr(a, k)), np.asarray(getattr(b, k)), rtol=0, atol=1e-15)
    assert (a.incidence != b.incidence).nnz == 0


def test_the_gradient_reaches_the_vertices():
    """dL/d(apex height) against a central difference, and it must have the right sign."""
    P = jnp.asarray(ARC, dtype=float)
    g = jax.grad(_inductance)(P)
    assert bool(jnp.all(jnp.isfinite(g)))
    h = 1e-7
    for i, j in ((1, 2), (2, 0)):  # the apex height, and the far end along the wire
        fd = (float(_inductance(P.at[i, j].add(h))) - float(_inductance(P.at[i, j].add(-h)))) / (2 * h)
        assert abs(float(g[i, j]) / fd - 1) < 1e-6
    assert float(g[1, 2]) > 0  # a taller loop encloses more area, so it has MORE inductance


def test_a_flatter_loop_really_is_less_inductive():
    """The gradient claims lowering the apex helps; check the value agrees over a real step."""
    tall = float(_inductance(jnp.asarray(ARC, dtype=float)))
    flat = float(_inductance(jnp.asarray([(0, 0, 0), (5 * MM, 0, 0.5 * MM), (10 * MM, 0, 0)], dtype=float)))
    assert flat < tall


def test_pair_matrix_is_differentiable_in_the_POSITIONS():
    """The regression this exposed: masking a singular kernel is not enough on its own.

    A same-element pair sits at zero separation and `d(sqrt)/dx` is infinite there, so `where`
    applied after the sqrt still differentiated the dead branch and returned NaN -- while the
    forward value stayed exactly right. Nothing but differentiating w.r.t. the coordinates shows it.
    """
    rng = np.random.default_rng(0)
    pos = jnp.asarray(rng.normal(size=(12, 3)))
    mom = jnp.asarray(rng.normal(size=(12, 3)))
    grp = np.repeat(np.arange(4), 3)
    sg = jnp.asarray(rng.uniform(1.0, 2.0, 4))

    def total(p):
        return jnp.sum(jnp.abs(pair_matrix(p, mom, lambda r: 1.0 / r, sg, group=grp)))

    g = jax.grad(total)(pos)
    assert bool(jnp.all(jnp.isfinite(g)))
    k = int(np.argmax(np.abs(np.asarray(g))))
    i, j = divmod(k, 3)
    h = 1e-6
    fd = (float(total(pos.at[i, j].add(h))) - float(total(pos.at[i, j].add(-h)))) / (2 * h)
    assert abs(float(g[i, j]) / fd - 1) < 1e-6


def test_the_wrong_number_of_point_arrays_is_refused():
    with pytest.raises(ValueError, match="point arrays for"):
        line_filaments([_shape(), _shape()], points=[jnp.asarray(ARC, dtype=float)])


# --------------------------------------------------------------------------------------------
# A wire that moves while it is WELDED to a fixed lattice -- the case a real package needs.
# --------------------------------------------------------------------------------------------


def _welded_case():
    """A trace layer with one wire arcing over it, welded where the wire lands."""
    from jno.peec import _weld
    from jno.utils.solver.peec import bar_filaments

    plate = jno.Shape.box(0, 0, 0, 20 * MM, 6 * MM, 1 * MM)
    fb = bar_filaments(plate, size=(2 * MM, 2 * MM, 1 * MM), sigma=[SIG])
    nb = len(np.asarray(fb.length))
    arc = [(3 * MM, 3 * MM, 0.6 * MM), (10 * MM, 3 * MM, 4 * MM), (17 * MM, 3 * MM, 0.6 * MM)]
    sh = jno.Shape.line(arc, r=RW, size=2 * MM)
    fl = line_filaments(sh)
    nl = len(np.asarray(fl.length))
    fil, sigma = _weld([(fb, fb.lattice["sigma"]), (fl, jnp.full(nl, SIG))], [[plate], [sh]])
    p = np.asarray(fil.nodes)
    term = {
        "A": terminal_nodes(fil, lambda q: q[:, 0] < p[:, 0].min() + 1e-9),
        "B": terminal_nodes(fil, lambda q: q[:, 0] > p[:, 0].max() - 1e-9),
    }
    return fil, sigma, term, sh, np.asarray(arc), nb


FIL, SIGMA, TERM2, SH, ARC2, NBAR = _welded_case()


def _welded_L(apex_z):
    """Loop inductance with the wire's apex raised to `apex_z`; the lattice never moves."""
    pts = jnp.asarray(ARC2).at[1, 2].set(apex_z)
    fl = line_filaments(SH, points=[pts])
    fil = FIL._replace(  # the geometry moves, the structure -- incidence, weld, numbering -- does not
        pos=jnp.concatenate([FIL.pos[: NBAR * 3], fl.pos]),
        mom=jnp.concatenate([FIL.mom[: NBAR * 3], fl.mom]),
        self_g=jnp.concatenate([FIL.self_g[:NBAR], fl.self_g]),
        length=jnp.concatenate([FIL.length[:NBAR], fl.length]),
    )
    _c, _p, inj = solve_network(
        fil, SIGMA, TERM2, [("A", "B", 1.0 + 0j)], (), (), (), omega=2 * np.pi * 1e6, matrix_free=True
    )
    return jnp.imag(1.0 / inj["A"]) / (2 * np.pi * 1e6)


def test_a_welded_wire_is_differentiable_in_its_shape():
    """The whole point: a bond wire being routed over a trace layer that stays where it is.

    Three places used to demand a CONCRETE geometry on the way to the answer, all of them on the
    preconditioner's side of the solve -- `lattice_apply` sampled element lengths and self terms out
    of the per-element arrays instead of reading the grid it already describes, `_lattice_diag`
    contracted the moments in numpy, and `near_block` built its neighbour list the same way. None of
    them needed to; the first two are lattice constants and the third only has to accelerate.
    """
    z0 = float(ARC2[1, 2])
    g = float(jax.grad(_welded_L)(z0))
    assert np.isfinite(g)
    h = 1e-6
    fd = (float(_welded_L(z0 + h)) - float(_welded_L(z0 - h))) / (2 * h)
    assert abs(g / fd - 1) < 1e-6  # measured 1.3e-08
    # NEGATIVE here, and that is not the bare-arc case above. This wire is welded at BOTH ends, so
    # it is a path in parallel with the plate rather than a loop over a return: 8.0352 nH at the
    # reference height, 7.9627 a millimetre higher. The sign is worth pinning precisely because it
    # is the opposite of the obvious one.
    assert g < 0


def test_only_the_wire_block_moves():
    """The lattice half of a welded network must be untouched by the wire being re-routed."""
    ref = np.asarray(FIL.length[:NBAR]).copy()
    moved = jnp.concatenate([FIL.length[:NBAR], line_filaments(SH, points=[jnp.asarray(ARC2).at[1, 2].add(2e-3)]).length])
    assert np.allclose(np.asarray(moved[:NBAR]), ref, rtol=0, atol=0)  # bit-identical, not merely close
    assert not np.allclose(np.asarray(moved[NBAR:]), np.asarray(FIL.length[NBAR:]))  # the wire did move


def test_wire_radius_is_a_design_variable():
    """A bond wire's GAUGE is a design variable, and a differentiable one.

    It enters the conducting area, the self term and the skin depth -- all of which were already
    jax -- so only the plumbing was missing. It matters because the realistic bond-wire question is
    not "thicker is better" (it is, trivially) but "given a fixed total cross-section, which wires
    should carry it": a wire that carries no power current is spending copper for nothing.
    """
    P = jnp.asarray(ARC, dtype=float)
    ref = line_filaments(_shape(), points=[P])

    same = line_filaments(_shape(), points=[P], radii=[RW])
    assert np.allclose(np.asarray(same.area), np.asarray(ref.area), rtol=0, atol=1e-18)
    assert np.allclose(np.asarray(same.self_g), np.asarray(ref.self_g), rtol=0, atol=1e-18)

    thick = line_filaments(_shape(), points=[P], radii=[2.0 * RW])
    assert np.allclose(np.asarray(thick.area), 4.0 * np.asarray(ref.area))  # area goes as r^2

    def loss(r):
        f = line_filaments(_shape(), points=[P], radii=[r])
        _c, _p, inj = solve_network(
            f, SIG, TERM, [("A", "B", 1.0 + 0j)], (), (), (), omega=2 * np.pi * 1e7, matrix_free=False
        )
        return jnp.imag(1.0 / inj["A"]) / (2 * np.pi * 1e7)

    g = float(jax.grad(loss)(RW))
    h = 1e-8
    fd = (float(loss(RW + h)) - float(loss(RW - h))) / (2 * h)
    assert np.isfinite(g)
    assert abs(g / fd - 1) < 1e-5
    assert g < 0  # a thicker wire has less self inductance


def test_the_wrong_number_of_radii_is_refused():
    with pytest.raises(ValueError, match="radii for"):
        line_filaments([_shape(), _shape()], radii=[RW])
