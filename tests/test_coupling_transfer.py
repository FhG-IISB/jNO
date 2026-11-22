"""Interface data between non-matching discretisations (jno._coupling.transfer).

Expectations come from geometry, not from the code: a linear field is reproduced exactly where both sides
describe the same plane or line; the weights integrate the measure exactly (a length, an area, the lateral
area 2πRH of a cylinder for the axisymmetric measure); a conservative map keeps Σ w q to round-off.
"""

import jax
import numpy as np
import pytest

import jno
from jno._coupling.transfer import Interface, conservative, consistent


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _linear(p):
    return 1.0 + 2.0 * p[:, 0] - 3.0 * p[:, 1] + (0.5 * p[:, 2] if p.shape[1] == 3 else 0.0)


def _pair_2d():
    a = Interface.of(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain(), "right")
    b = Interface.of(jno.shape.rect(1.0, 0.0, 2.0, 1.0, size=0.07).domain(), "left")
    return a, b


def _pair_3d():
    a = Interface.of(jno.shape.box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, size=0.3).domain(), "right")
    b = Interface.of(jno.shape.box(1.0, 0.0, 0.0, 2.0, 1.0, 1.0).structured(n=5).quad().domain(), "left")
    return a, b


@pytest.mark.parametrize("pair", [_pair_2d, _pair_3d], ids=["2d-segments", "3d-triangles-quads"])
def test_consistent_map_reproduces_a_linear_field_on_a_shared_plane(pair):
    a, b = pair()
    assert a.points.shape[0] != b.points.shape[0]  # the meshes do not match
    for t, s in ((a, b), (b, a)):
        np.testing.assert_allclose(consistent(t, s) @ _linear(s.points), _linear(t.points), atol=1e-12)


def test_weights_integrate_the_measure():
    a, b = _pair_2d()
    assert a.total == pytest.approx(1.0, abs=1e-13) and b.total == pytest.approx(1.0, abs=1e-13)
    a3, b3 = _pair_3d()
    assert a3.total == pytest.approx(1.0, abs=1e-12) and b3.total == pytest.approx(1.0, abs=1e-12)
    # the line x = 1, 0 <= r <= 1 of an (x, r) meridian plane swept about the axis: a disk of area π
    axi = Interface.of(jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain(), "right", measure="axisymmetric")
    assert axi.total == pytest.approx(np.pi, abs=1e-12)
    # the wall r = 0.5, 0 <= x <= 2 swept about the axis: a cylinder of lateral area 2π R L
    wall = Interface.of(jno.shape.rect(0.0, 0.0, 2.0, 0.5, size=0.1).domain(), "top", measure="axisymmetric")
    assert wall.total == pytest.approx(2.0 * np.pi * 0.5 * 2.0, abs=1e-12)


@pytest.mark.parametrize("measure", [None, "axisymmetric"])
def test_conservative_map_keeps_the_total_on_a_curved_interface(measure):
    # two discretisations of the same circle (shifted off the axis so 2πr stays positive)
    a = Interface.of(jno.shape.disk(0.0, 2.0, 1.0, size=0.1).domain(), "boundary", measure=measure)
    b = Interface.of(jno.shape.disk(0.0, 2.0, 1.0, size=0.063).domain(), "boundary", measure=measure)
    q = np.random.default_rng(0).normal(size=a.points.shape[0])
    for t, s, qs in ((b, a, q), (a, b, consistent(b, a) @ q)):
        qt = conservative(t, s) @ qs
        assert float(t.weights @ qt) == pytest.approx(float(s.weights @ qs), rel=1e-13, abs=1e-13)


def test_consistent_map_converges_at_second_order_on_a_curved_interface():
    # a smooth field read across two discretisations of the same circle: the error is the chord gap, O(h^2)
    def err(h):
        a = Interface.of(jno.shape.disk(0.0, 0.0, 1.0, size=h).domain(), "boundary")
        b = Interface.of(jno.shape.disk(0.0, 0.0, 1.0, size=0.63 * h).domain(), "boundary")
        f = lambda p: np.cos(np.arctan2(p[:, 1], p[:, 0]))  # noqa: E731
        return np.abs(consistent(b, a) @ f(a.points) - f(b.points)).max()

    e1, e2 = err(0.2), err(0.1)
    assert 1.6 < np.log2(e1 / e2) < 2.6
