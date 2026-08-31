"""``sol.field(points)`` -- the magnetic flux density the solved currents produce, in free space.

A partial-element method never meshes the air, so the field off the metal is not a solved unknown:
it is a Biot-Savart sum over the currents that WERE solved for. That makes it a readout rather than
a second problem, and it is differentiable in the currents and in the evaluation points alike.

The oracles here are closed form, and deliberately not the solver's own machinery:

  infinite straight wire    B = mu0 I / (2 pi r),   approached by a long finite one
  circular loop, on axis    B = mu0 I R^2 / (2 (R^2 + z^2)^(3/2))       (Biot-Savart, exact)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.peec import line_filaments, solve_network, terminal_nodes

jax.config.update("jax_enable_x64", True)

MU0 = 4e-7 * np.pi
SIG = 5.8e7


def _straight(ell=2.0, n=400, a=1e-3):
    """A long straight wire carrying 1 A, so the infinite-wire law applies near its middle."""
    sh = jno.Shape.line([(0, 0, -ell / 2), (0, 0, ell / 2)], r=a, size=ell / n)
    f = line_filaments(sh)
    p = np.asarray(f.nodes)
    term = {
        "A": terminal_nodes(f, lambda q: q[:, 2] < p[:, 2].min() + 1e-9),
        "B": terminal_nodes(f, lambda q: q[:, 2] > p[:, 2].max() - 1e-9),
    }
    return f, term


def _solve(f, term):
    cur, _phi, inj = solve_network(f, SIG, term, [("A", "B", 1.0 + 0j)], (), (), (), omega=0.0, matrix_free=False)
    return cur, complex(inj["A"])


def test_a_long_straight_wire_gives_the_infinite_wire_law():
    """B = mu0 I / (2 pi r) around the middle of a long wire, and it must fall as 1/r."""
    from jno.peec import PEECSolution

    f, term = _straight()
    cur, Ip = _solve(f, term)
    sol = PEECSolution(0.0, cur, f, {}, 1.0 / Ip, np.zeros(1))
    for r in (0.01, 0.02, 0.05):
        B = np.asarray(sol.field(np.array([[r, 0.0, 0.0]])))[0]
        exact = MU0 * abs(Ip) / (2 * np.pi * r)
        assert abs(np.linalg.norm(B) / exact - 1) < 2e-3, f"r={r}: {np.linalg.norm(B)} vs {exact}"


def test_the_field_of_a_z_directed_current_curls_around_it():
    """Direction, not just magnitude: B is azimuthal, so B . r_hat and B . z_hat both vanish."""
    from jno.peec import PEECSolution

    f, term = _straight()
    cur, Ip = _solve(f, term)
    sol = PEECSolution(0.0, cur, f, {}, 1.0 / Ip, np.zeros(1))
    B = np.asarray(sol.field(np.array([[0.02, 0.0, 0.0]])))[0]
    assert abs(B[0]) < 1e-3 * np.linalg.norm(B)  # nothing radial
    assert abs(B[2]) < 1e-3 * np.linalg.norm(B)  # nothing along the wire
    assert abs(B[1]) > 0.99 * np.linalg.norm(B)  # all azimuthal


def test_an_arc_on_its_axis_matches_the_closed_form():
    """B_z = (theta / 2 pi) * mu0 I R^2 / (2 (R^2 + z^2)^{3/2}), on the axis, at several heights.

    A CLOSED loop cannot carry a port current -- both terminals would be the same point and the
    source is shorted -- so this is the standard nearly-closed arc, one segment short of a circle,
    and the closed form is scaled by the fraction of the turn that is actually there. That factor is
    exact on the axis, where every element contributes in proportion to the angle it subtends.
    """
    from jno.peec import PEECSolution

    R, n = 0.05, 256
    th = np.linspace(0, 2 * np.pi * (1 - 1.0 / n), n)  # one segment short: the gap the port sits in
    frac = 1.0 - 1.0 / n
    pts = [(R * np.cos(t), R * np.sin(t), 0.0) for t in th]
    sh = jno.Shape.line(pts, r=5e-4, size=2 * np.pi * R / n)
    f = line_filaments(sh)
    nd = np.asarray(f.nodes)
    term = {
        "A": terminal_nodes(f, lambda q: np.linalg.norm(q - nd[0], axis=1) < 1e-12),
        "B": terminal_nodes(f, lambda q: np.linalg.norm(q - nd[-1], axis=1) < 1e-12),
    }
    cur, _phi, inj = solve_network(f, SIG, term, [("A", "B", 1.0 + 0j)], (), (), (), omega=0.0, matrix_free=False)
    Ip = complex(inj["A"])
    sol = PEECSolution(0.0, cur, f, {}, 1.0 / Ip, np.zeros(1))
    for z in (0.0, 0.02, 0.10):
        B = np.asarray(sol.field(np.array([[0.0, 0.0, z]])))[0]
        exact = frac * MU0 * abs(Ip) * R**2 / (2 * (R**2 + z**2) ** 1.5)
        assert abs(abs(B[2]) / exact - 1) < 5e-3, f"z={z}: {B[2]} vs {exact}"


def test_the_field_is_differentiable_in_the_EVALUATION_POINT():
    """A gradient in the probe position, which is what an EMI objective on a keep-out zone needs."""
    from jno.peec import PEECSolution

    f, term = _straight()
    cur, Ip = _solve(f, term)
    sol = PEECSolution(0.0, cur, f, {}, 1.0 / Ip, np.zeros(1))

    def mag(r):
        return jnp.linalg.norm(sol.field(jnp.array([[r, 0.0, 0.0]]))[0])

    g = float(jax.grad(mag)(0.02))
    h = 1e-7
    fd = (float(mag(0.02 + h)) - float(mag(0.02 - h))) / (2 * h)
    assert np.isfinite(g) and abs(g / fd - 1) < 1e-5
    assert g < 0  # 1/r: the field falls as the probe moves away


def test_many_points_at_once_agree_with_one_at_a_time():
    """It is a sum over elements per point, so batching must not change any of them."""
    from jno.peec import PEECSolution

    f, term = _straight()
    cur, Ip = _solve(f, term)
    sol = PEECSolution(0.0, cur, f, {}, 1.0 / Ip, np.zeros(1))
    pts = np.array([[0.01, 0.0, 0.0], [0.0, 0.03, 0.01], [0.02, 0.02, -0.05]])
    together = np.asarray(sol.field(pts))
    apart = np.stack([np.asarray(sol.field(p[None]))[0] for p in pts])
    assert np.allclose(together, apart, rtol=0, atol=1e-18)


def test_a_point_INSIDE_the_metal_is_refused_rather_than_returning_a_number():
    """The kernel is singular on a filament, and the field inside metal is not what this computes.

    Note the point has to be inside the CROSS-SECTION to be wrong: on the AXIS of a straight
    filament the field is zero by symmetry, not infinite, so an axis point would not catch anything.
    """
    from jno.peec import PEECSolution

    f, term = _straight(a=1e-3)
    cur, Ip = _solve(f, term)
    sol = PEECSolution(0.0, cur, f, {}, 1.0 / Ip, np.zeros(1))
    inside = np.asarray(f.pos)[0] + np.array([0.4e-3, 0.0, 0.0])  # 0.4 mm off a 1 mm-radius wire
    with pytest.raises(ValueError, match="inside|singular"):
        sol.field(inside[None])
