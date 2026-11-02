"""Surface terms on INTERIOR interfaces (a tag such as "gas|solid" between two regions).

They used to integrate over nothing, silently: the assembler took surface facets from the outer boundary only.
Oracles: the interface length; and a slab with a heat sink h T on an interior interface at x = a, T = 1 at both
ends, k = 1: T is piecewise linear with T(a) = 1 / (1 + h a (1 - a)), which P1 represents exactly."""

import numpy as np
import pytest
from jax import config

import jno

config.update("jax_enable_x64", True)
A, H = 0.4, 2.0


def _slab(h=0.05):
    R = jno.shape.rect
    d = (R(0, 0, A, 0.1).name("solid") + R(0, 0, 1, 0.1).name("gas")).sized(h).domain()
    T, v = d.fem_symbols(names=("T", "v"))
    x, y, _ = d.variable("interior", split=True)
    iface = next(k for k in d.mesh.cell_sets if "|" in k)
    xs, ys, _ = d.variable(iface, split=True)
    return d, T, v, x, y, T.bind(x=xs, y=ys), v.bind(x=xs, y=ys)


def test_a_surface_term_on_an_interior_interface_integrates_over_it():
    d, T, v, x, y, Ts, vs = _slab()
    Ti, vi = T.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem([Ti.d(x) * vi.d(x) + Ti.d(y) * vi.d(y), 1.0 * vs])
    assert np.asarray(fem.residual(np.zeros(len(d.mesh.points)))).sum() == pytest.approx(0.1, rel=1e-12)


def test_a_heat_sink_on_an_interior_interface_matches_the_exact_profile():
    d, T, v, x, y, Ts, vs = _slab()
    Ti, vi = T.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem(
        [
            Ti.d(x) * vi.d(x) + Ti.d(y) * vi.d(y),
            H * Ts * vs,
            T(*d.variable("left", split=True)[:2]) - 1.0,
            T(*d.variable("right", split=True)[:2]) - 1.0,
        ]
    )
    sol = np.asarray(fem.solve(linear=jno.solve.lu()))
    p = np.asarray(d.mesh.points)[:, 0]
    Ta = 1.0 / (1.0 + H * A * (1 - A))
    exact = np.where(p <= A, 1 + (Ta - 1) * p / A, 1 + (Ta - 1) * (1 - p) / (1 - A))
    np.testing.assert_allclose(sol, exact, atol=1e-9)


def test_interface_facets_stay_out_of_the_boundary_catch_all():
    d, T, v, x, y, Ts, vs = _slab()
    Ti, vi = T.bind(x=x, y=y), v.bind(x=x, y=y)
    xb, yb, _ = d.variable("boundary", split=True)
    vb = v.bind(x=xb, y=yb)
    fem = jno.fem([Ti.d(x) * vi.d(x) + Ti.d(y) * vi.d(y), 1.0 * vb, 0.0 * Ts * vs])  # interface term present too
    assert np.asarray(fem.residual(np.zeros(len(d.mesh.points)))).sum() == pytest.approx(2.2, rel=1e-12)
