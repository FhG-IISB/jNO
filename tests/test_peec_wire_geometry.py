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
