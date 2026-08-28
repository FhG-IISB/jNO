"""Moving a FEATURE on a fixed lattice, by gradient.

The bar lattice cannot move: its occupancy is a discrete in/out test per cell, and the FFT needs a
regular grid. So the geometry is not carried by the mesh at all -- it is projected onto the density
field as a smooth indicator of the shape's signed distance, and the conductivity follows. The
lattice never changes, the FFT structure is untouched, and a feature can be positioned by gradient
including by less than one cell.

Norato, Bell & Tortorelli, "A geometry projection method for continuum-based topology optimization",
Comput. Methods Appl. Mech. Engrg. 293 (2015), sec. 2.

This pins a USAGE contract rather than a function: that `solve_network` takes a traced per-element
conductivity and differentiates it, which is the whole mechanism.
"""

import jax
import jax.numpy as jnp
import numpy as np

import jno
from jno.utils.solver.peec import bar_filaments, solve_network

jax.config.update("jax_enable_x64", True)

SIG, MM, H, RV = 5.8e7, 1e-3, 2e-3, 5e-3


def _plate():
    f = bar_filaments(jno.Shape.box(0, 0, 0, 30 * MM, 20 * MM, 2 * MM), size=(H, H, 2 * MM))
    node = np.asarray(f.nodes)
    csc = f.incidence.tocsc()
    ne = len(np.asarray(f.length))
    pair = np.array([csc.indices[csc.indptr[k] : csc.indptr[k + 1]][:2] for k in range(ne)])
    lo, hi = node.min(0), node.max(0)
    term = {"A": np.flatnonzero(node[:, 0] < lo[0] + H), "B": np.flatnonzero(node[:, 0] > hi[0] - H)}
    return f, jnp.asarray(node), pair, term


F, C, PAIR, TERM = _plate()


def _resistance(rho):
    cs = SIG * (1e-4 + (1 - 1e-4) * rho**3)
    bs = 0.5 * (cs[PAIR[:, 0]] + cs[PAIR[:, 1]])
    _c, _p, inj = solve_network(F, bs, TERM, [("A", "B", 1.0 + 0j)], (), (), (), omega=0.0, matrix_free=True)
    return jnp.real(1.0 / inj["A"])


def soft(theta, eps=0.6 * H):
    """A void of radius RV centred at theta, blended over eps -- 1 outside it, 0 inside."""
    d = jnp.sqrt(jnp.sum((C[:, :2] - jnp.asarray(theta)[None, :]) ** 2, axis=1) + 1e-30)
    return _resistance(jax.nn.sigmoid((d - RV) / eps))


def mask(theta):
    """The same void as an in/out containment test, which is what a mesh-based mask gives."""
    d = np.sqrt(np.sum((np.asarray(C)[:, :2] - np.asarray(theta)[None, :]) ** 2, axis=1))
    return (d > RV).astype(float)


def hard(theta):
    return float(_resistance(jnp.asarray(mask(theta))))


OFF = np.array([15 * MM, 7 * MM])  # off the symmetry axis, where dR/dy is not zero


def test_the_projected_gradient_converges_to_the_finite_difference():
    """AD is exact, so shrinking the difference step must walk towards it, not away."""
    g = float(jax.grad(lambda t: soft(t))(jnp.asarray(OFF))[1])
    assert abs(g) > 1e-6  # a real sensitivity, not a numerical zero
    rel = []
    for step in (0.5 * H, 0.05 * H, 0.01 * H):
        d = np.array([0.0, step])
        fd = (float(soft(OFF + d)) - float(soft(OFF - d))) / (2 * step)
        rel.append(abs(g / fd - 1))
    assert rel[0] > rel[1] > rel[2]  # converging
    assert rel[2] < 1e-3


def test_a_hard_mask_cannot_give_the_derivative_at_all():
    """The reason projection is needed: differencing a staircase is not a derivative.

    A containment test changes only when a cell CENTRE crosses the boundary, so the response is flat
    between jumps. Measured here: at half a cell the difference is 2.6x the true slope, and at a
    hundredth of a cell it is EXACTLY ZERO -- the whole step sits inside one tread. There is no step
    size that recovers the derivative, which is the point; it is not a resolution problem.
    """
    truth = float(jax.grad(lambda t: soft(t))(jnp.asarray(OFF))[1])
    coarse = (hard(OFF + np.array([0.0, 0.5 * H])) - hard(OFF - np.array([0.0, 0.5 * H]))) / H
    assert abs(coarse / truth - 1) > 0.5  # far off at half a cell
    for frac in (0.05, 0.01):  # and refining does not help: it lands on a flat
        d = np.array([0.0, frac * H])
        # the structural fact: not one cell changes hands, so the two solves pose the SAME problem
        assert np.array_equal(mask(OFF + d), mask(OFF - d))
        fd = (hard(OFF + d) - hard(OFF - d)) / (2 * frac * H)
        assert abs(fd) < 1e-6 * abs(truth)  # what is left is the solver's own round-off


def test_projection_makes_sub_cell_motion_visible():
    """Every tenth of a cell moves the answer, monotonically; a mask repeats itself instead."""
    ts = np.linspace(0, 1, 6)
    vals = [float(soft(OFF + np.array([0.0, t * H]))) for t in ts]
    assert all(b < a for a, b in zip(vals, vals[1:]))  # monotone, matching the negative gradient
    assert len({round(v, 12) for v in vals}) == len(vals)  # and every step is distinct
    masked = [hard(OFF + np.array([0.0, t * H])) for t in ts]
    assert len({round(v, 10) for v in masked}) < len(masked)  # the staircase repeats values
