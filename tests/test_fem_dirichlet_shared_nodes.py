"""A node on two Dirichlet regions -- the corner where ``left`` meets ``bottom`` -- is named by both.

The elimination appended one unit-diagonal triplet per (dof, value) pair, and BCOO sums duplicates, so the
corner's row became ``2 s u = s g``: the corner solved to HALF its prescribed value, with nothing reported
(``u = 1 + y`` on all four edges gave 0.5 and 1.0 at the corners, against 1 and 2). The pairs are now one
per DOF; two different values at one DOF keep the later condition and say so.

Oracle: ``-Δu = 0`` with ``u = 1 + y`` on the boundary is solved exactly by P1 (the solution is linear).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

J = jno.np


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _laplace(edges, value, *, nonlinear=False):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=2).domain()
    x, y = d.variable("interior", split=True)[:2]
    u = d.unknown()
    v = u.test()
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    k = (1.0 + 0.0 * ui**2) if nonlinear else 1.0  # nonlinear in form only: routes to the Newton path
    terms = [k * (ui.x * vi.x + ui.y * vi.y)]
    for e in edges:
        xe, ye = d.variable(e, split=True)[:2]
        terms.append(u(xe, ye) - value(xe, ye))
    return jno.fem(terms)


@pytest.mark.parametrize("nonlinear", [False, True])
@pytest.mark.parametrize("edges", [("left", "bottom"), ("left", "right", "bottom", "top")])
def test_a_corner_on_two_regions_takes_its_value(edges, nonlinear):
    fem = _laplace(edges, lambda x, y: 1.0 + y, nonlinear=nonlinear)
    kw = {"nonlinear": jno.solve.newton(direct=True, rtol=1e-12, atol=1e-13)} if nonlinear else {"linear": jno.solve.lu()}
    U = np.asarray(fem.solve(**kw)).reshape(-1)
    P = np.asarray(fem.points)
    on = np.zeros(len(P), dtype=bool)
    for e in edges:
        on |= {"left": P[:, 0] < 1e-9, "right": P[:, 0] > 1 - 1e-9, "bottom": P[:, 1] < 1e-9, "top": P[:, 1] > 1 - 1e-9}[e]
    np.testing.assert_allclose(U[on], 1.0 + P[on, 1], atol=1e-12)
    if len(edges) == 4:
        np.testing.assert_allclose(U, 1.0 + P[:, 1], atol=1e-12)


def test_two_different_values_at_a_corner_keep_the_later_one_and_say_so(capfd):
    """A lid-driven cavity's top corners: the walls say 0, the lid says 1. The lid is written last."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0).structured(n=2).domain()
    x, y = d.variable("interior", split=True)[:2]
    u = d.unknown()
    v = u.test()
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    terms = [ui.x * vi.x + ui.y * vi.y]
    for e in ("left", "right", "bottom"):
        xe, ye = d.variable(e, split=True)[:2]
        terms.append(u(xe, ye) - 0.0)
    xt, yt = d.variable("top", split=True)[:2]
    terms.append(u(xt, yt) - 1.0)
    fem = jno.fem(terms)
    out, err = capfd.readouterr()
    assert "DIFFERENT values" in out + err
    U = np.asarray(fem.solve(linear=jno.solve.lu())).reshape(-1)
    P = np.asarray(fem.points)
    top = P[:, 1] > 1 - 1e-9
    np.testing.assert_allclose(U[top], 1.0, atol=1e-12)  # the corners too: the lid was written last
