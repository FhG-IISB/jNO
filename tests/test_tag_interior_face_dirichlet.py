"""A Dirichlet condition on a ``d.tag(name, where, region=body)`` that lies INSIDE a conforming mesh.

Such a tag names part of a body's surface where it meets another body (a piece of an interface).
Surface terms on it were already integrated over those interior facets, but the Dirichlet node lookup
walked the outer boundary only: ``T(seg) - 1`` pinned just the one segment node that also lies on the
outer boundary and left the other ten free, with no warning (measured on the stacked squares below:
the interface came back at 0.41-0.73 instead of 1).
"""

from __future__ import annotations

import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    import jax

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _solve(where):
    """Two stacked unit squares (a below b), T = 0 on the bottom, T = 1 on the tagged interface piece."""
    d = (jno.shape.rect(0, 0, 1, 1).name("a") + jno.shape.rect(0, 1, 1, 2).name("b")).sized(0.1).domain()
    d.tag("seg", where, region="a")
    T = d.unknown.scalar("T")
    v = T.test()
    x, y, _ = d.variable("interior", split=True)
    xs, ys, _ = d.variable("seg", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    Ti, vi = T.bind(x=x, y=y), v.bind(x=x, y=y)
    fem = jno.fem([Ti.d(x) * vi.d(x) + Ti.d(y) * vi.d(y), T(xs, ys) - 1.0, T(xb, yb) - 0.0])
    u = np.asarray(fem.solve(linear=jno.solve.lu())).ravel()
    return np.asarray(d.mesh.points)[:, :2], u


def test_a_pin_on_the_whole_interface_given_by_a_predicate_is_exact():
    """Insulated sides and top: T = y in a, T = 1 in b, which P1 reproduces exactly."""
    p, u = _solve(lambda x, y: np.abs(y - 1.0) < 1e-9)
    exact = np.minimum(p[:, 1], 1.0)
    assert np.abs(u - exact).max() < 1e-9


def test_a_pin_on_part_of_the_interface_reaches_every_node_of_that_part_and_no_other():
    p, u = _solve(lambda x, y: (np.abs(y - 1.0) < 1e-9) & (x < 0.5 + 1e-9))
    on_iface = np.abs(p[:, 1] - 1.0) < 1e-9
    inside = on_iface & (p[:, 0] < 0.5 + 1e-9)
    assert inside.sum() >= 6
    assert np.abs(u[inside] - 1.0).max() < 1e-12
    assert u[on_iface & (p[:, 0] > 0.5 + 1e-9)].max() < 1.0 - 1e-3  # the rest of the interface stays free
