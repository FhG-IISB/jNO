"""Guards for :mod:`jno.utils.solver.peec` — geometry to PEEC filaments.

Both physics tests go from a ``Shape`` all the way to a closed form, so they cover the
discretisation, the self term and the kernel together:

* a straight wire, ``L = (mu0 l/2pi)[ln(2l/a) - 3/4]``
* a circular loop, ``L = mu0 R [ln(8R/a) - 7/4]``

Neither is exact at any finite quadrature order, and that is the point: the residual IS the
near-field quadrature, and ``test_loop_converges_in_quadrature_order`` pins that it converges rather
than sitting at some floor. Measured on a 128-sided loop: 10.5 % at one Gauss point per filament,
3.4 % at two, 1.7 % at three, 0.31 % at eight, 0.16 % at twelve.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.kernel import pair_quadratic
from jno.utils.solver.peec import line_filaments

MU0 = 4e-7 * np.pi
INV_R = lambda r: 1.0 / r  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _inductance(shape, size, quad, chunk=256):
    pos, mom, self_g, group = line_filaments(shape, size=size, quad=quad)
    q = pair_quadratic(pos, mom, INV_R, self_g, group=group, chunk=chunk)
    return float(q) * MU0 / (4 * np.pi)


def _circle(R, n):
    th = np.linspace(0.0, 2 * np.pi, n + 1)
    return [tuple(p) for p in np.stack([R * np.cos(th), R * np.sin(th), np.zeros_like(th)], -1)]


def test_straight_wire_matches_its_closed_form():
    """``L = (mu0 l/2pi)[ln(2l/a) - 3/4]``, including the internal inductance of a uniform current."""
    L, a = 0.050, 2.5e-4
    exact = (MU0 * L / (2 * np.pi)) * (np.log(2 * L / a) - 0.75)
    got = _inductance(jno.Shape.line([(0, 0, 0), (0, 0, L)], r=a), size=L / 16, quad=3)
    assert abs(got / exact - 1) < 0.02, f"{got * 1e9:.4f} nH vs {exact * 1e9:.4f} nH"


def test_loop_converges_in_quadrature_order():
    """A circular loop against ``mu0 R [ln(8R/a) - 7/4]``, refining the QUADRATURE not the mesh.

    Subdividing the loop further does not help -- the error lives in the near-field mutual, not in
    the discretisation -- so this refines ``quad`` and requires monotone convergence.
    """
    R, a, ns = 0.010, 2.5e-4, 128
    exact = MU0 * R * (np.log(8 * R / a) - 1.75)
    wire = jno.Shape.line(_circle(R, ns), r=a)
    err = [abs(_inductance(wire, 2 * np.pi * R / ns, q) / exact - 1) for q in (1, 2, 3, 8)]
    assert err[0] > 0.05, f"one point per filament should be several percent low, got {err[0]:.4f}"
    assert all(b < a_ for a_, b in zip(err, err[1:])), f"not monotone in quadrature order: {err}"
    assert err[-1] < 0.01, f"eight Gauss points should be under 1%, got {err[-1]:.4f}"


def test_refining_the_polyline_does_not_fix_the_near_field():
    """Records WHY the accuracy knob is ``quad`` and not ``size``.

    More filaments at one Gauss point each converges to a WRONG value, because the error is in the
    mutual between neighbours rather than in the representation of the curve.
    """
    R, a = 0.010, 2.5e-4
    exact = MU0 * R * (np.log(8 * R / a) - 1.75)
    errs = [
        abs(_inductance(jno.Shape.line(_circle(R, n), r=a), 2 * np.pi * R / n, quad=1) / exact - 1) for n in (64, 128, 256)
    ]
    assert min(errs) > 0.05, f"refining alone should stay several percent out, got {errs}"


def test_filament_lengths_sum_to_the_polyline_length():
    pts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0), (1.0, 2.0, 0.5)]
    wire = jno.Shape.line(pts, r=0.01)
    _, mom, self_g, group = line_filaments(wire, size=0.17, quad=2)
    per = jax.ops.segment_sum(jnp.asarray(mom), jnp.asarray(group), num_segments=len(self_g))
    total = float(jnp.linalg.norm(per, axis=1).sum())
    exact = float(np.linalg.norm(np.diff(np.asarray(pts), axis=0), axis=1).sum())
    assert abs(total / exact - 1) < 1e-12


def test_a_vertex_is_always_a_filament_boundary():
    """A bend must not fall inside a straight element, even when ``size`` exceeds a whole leg."""
    pts = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.3, 0.9, 0.0)]
    _, mom, self_g, group = line_filaments(jno.Shape.line(pts, r=0.01), size=5.0, quad=1)
    per = np.asarray(jax.ops.segment_sum(jnp.asarray(mom), jnp.asarray(group), num_segments=len(self_g)))
    tang = per / np.linalg.norm(per, axis=1)[:, None]
    ok = [
        (np.abs(t - np.array([1.0, 0, 0])).max() < 1e-12) or (np.abs(t - np.array([0, 1.0, 0])).max() < 1e-12) for t in tang
    ]
    assert all(ok), f"a filament straddles the corner: {tang}"


def test_refuses_geometry_it_cannot_discretise():
    with pytest.raises(NotImplementedError, match="expected a single Line primitive"):
        line_filaments(jno.Shape.box(0, 0, 0, 1, 1, 1, size=0.5), size=0.5)


def test_refuses_to_guess_a_filament_length():
    with pytest.raises(ValueError, match="no filament length"):
        line_filaments(jno.Shape.line([(0, 0, 0), (0, 0, 1)], r=0.01))
