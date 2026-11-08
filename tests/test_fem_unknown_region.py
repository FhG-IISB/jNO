"""An unknown that lives on ONE region: ``d.unknown(..., region="fluid")``.

A fluid flowing inside a solid that otherwise only conducts heat: the velocity and the pressure exist in
the fluid, the temperature everywhere. Without a region every field carries DOFs over the whole mesh, and
those outside the fluid appear in no term -- the system is structurally singular (``test_fem_dead_dofs``).

``region=`` pins every DOF of the field on a node outside the region to 0. A pinned DOF is an identity row
with no off-diagonal entry, so it causes no fill-in: the factorisation is that of the region's own
system. Nodes on the region's boundary belong to it (a cell on each side), so they stay free and take
the conditions the problem gives them (a natural one, or a wall ``u(...) - 0``). ``p.pin()`` on such a
field picks its gauge vertex inside the region.

Oracles: P2 reproduces a quadratic exactly, and P1 pressure a linear hydrostatic one.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _stacked(h=0.25):
    """`lower` = [0,1]x[0,1] under `upper` = [0,1]x[1,2]."""
    return (jno.shape.rect(0, 0, 1, 1).name("lower") + jno.shape.rect(0, 1, 1, 2).name("upper")).sized(h).domain()


def _lin(fem, sol):
    return np.asarray(sol).reshape(-1)


def test_a_region_field_solves_on_its_region_and_is_zero_outside():
    """-u'' = 1 on `lower`, u = 0 at y = 0, natural at the interface y = 1: u = y - y^2/2, exact in P2.

    The interface nodes belong to `lower` and must stay free (the natural condition there gives
    u(1) = 1/2); every node strictly inside `upper` is 0."""
    d = _stacked()
    u = d.unknown.scalar(order=2, region="lower")
    v = u.test()
    x, y = d.variable("lower", split=True)[:2]
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    xb, yb = d.variable("bottom", split=True)[:2]
    fem = jno.fem([ub.x * vb.x + ub.y * vb.y - vb, u(xb, yb) - 0.0])
    sol = _lin(fem, fem.solve(linear=jno.solve.lu(backend="host")))
    pts = np.asarray(fem.points)
    inside = pts[:, 1] <= 1.0 + 1e-12
    exact = pts[:, 1] - 0.5 * pts[:, 1] ** 2
    np.testing.assert_allclose(sol[inside], exact[inside], atol=1e-10)
    assert np.all(sol[~inside] == 0.0)
    on_iface = np.abs(pts[:, 1] - 1.0) < 1e-12
    np.testing.assert_allclose(sol[on_iface], 0.5, atol=1e-10)


def test_the_pins_add_no_fill_in():
    """The region field factorises like the region alone: the pinned DOFs are rows with one entry."""
    import scipy.sparse as sp

    d = _stacked(h=0.1)
    u = d.unknown.scalar(order=2, region="lower")
    v = u.test()
    x, y = d.variable("lower", split=True)[:2]
    ub, vb = u.bind(x=x, y=y), v.bind(x=x, y=y)
    xb, yb = d.variable("bottom", split=True)[:2]
    fem = jno.fem([ub.x * vb.x + ub.y * vb.y - vb, u(xb, yb) - 0.0])
    A = fem.operator[0]
    idx = np.asarray(A.indices)
    M = sp.coo_matrix((np.asarray(A.data), (idx[:, 0], idx[:, 1])), shape=A.shape).tocsr()
    M.sum_duplicates()
    M.eliminate_zeros()
    pts = np.asarray(fem.points)
    outside = pts[:, 1] > 1.0 + 1e-12
    rows = np.diff(M.indptr)
    assert outside.sum() > 0
    assert np.all(rows[outside] == 1), "a DOF outside the region is coupled to something"
    assert np.all(M.diagonal()[outside] != 0.0)  # (scaled like the matrix diagonal, not 1)


def test_stokes_at_rest_on_a_region_gives_the_hydrostatic_pressure():
    """Taylor-Hood on `upper` only, walls all round it, gravity (0, -g): u = 0 and p = -g y + c.

    Both u and p are region fields. `upper` does not contain the mesh's min-corner, which is where
    `p.pin()` gauges by default: there the pin would land on a DOF already held at 0, leave the
    pressure's constant free and the saddle system singular. It must pick a vertex of `upper`."""
    g = 3.0
    d = _stacked()
    u = d.unknown.vector(2, order=2, region="upper")
    p = d.unknown.scalar(region="upper")
    v, q = u.test(), p.test()
    x, y = d.variable("upper", split=True)[:2]
    ub, vb, pb, qb = u.bind(x=x, y=y), v.bind(x=x, y=y), p.bind(x=x, y=y), q.bind(x=x, y=y)
    div_u = ub.x[0] + ub.y[1]
    div_v = vb.x[0] + vb.y[1]
    visc = ub.x[0] * vb.x[0] + ub.y[0] * vb.y[0] + ub.x[1] * vb.x[1] + ub.y[1] * vb.y[1]
    walls = [u(*d.variable(t, split=True)[:2]) - (0.0, 0.0) for t in ("top", "lower|upper")]
    xl, yl = d.variable("wl", where=lambda x_, y_: (x_ < 1e-9) & (y_ >= 1.0 - 1e-9), split=True)[:2]
    xr, yr = d.variable("wr", where=lambda x_, y_: (x_ > 1 - 1e-9) & (y_ >= 1.0 - 1e-9), split=True)[:2]
    walls += [u(xl, yl) - (0.0, 0.0), u(xr, yr) - (0.0, 0.0)]
    fem = jno.fem([visc - pb * div_v + g * vb[1], -qb * div_u, *walls, p.pin()])
    # The singular system still factorises (SuperLU takes the tiny pivot) and gives a finite, plausible
    # answer, so the rank is checked directly.
    A = fem.operator[0]
    idx = np.asarray(A.indices)
    dense = np.zeros(A.shape)
    np.add.at(dense, (idx[:, 0], idx[:, 1]), np.asarray(A.data))
    assert np.linalg.matrix_rank(dense) == A.shape[0], "the pressure gauge did not land in the region"
    sol = _lin(fem, fem.solve(linear=jno.solve.lu(backend="host")))
    offs = fem.offsets
    up, pp = sol[offs[0] : offs[1]], sol[offs[1] : offs[2]]
    np.testing.assert_allclose(up, 0.0, atol=1e-10)
    ppts = np.asarray(fem.field_points[1])
    inside = ppts[:, 1] >= 1.0 - 1e-12
    c = np.mean(pp[inside] + g * ppts[inside, 1])
    np.testing.assert_allclose(pp[inside], -g * ppts[inside, 1] + c, atol=1e-9)
    assert np.all(pp[~inside] == 0.0)


def test_an_unknown_region_is_refused_by_name():
    d = _stacked()
    u = d.unknown.scalar(region="nowhere")
    v = u.test()
    x, y = d.variable("lower", split=True)[:2]
    xb, yb = d.variable("bottom", split=True)[:2]
    with pytest.raises(ValueError, match="nowhere"):
        jno.fem([u.bind(x=x, y=y).x * v.bind(x=x, y=y).x - v.bind(x=x, y=y), u(xb, yb) - 0.0])


def test_a_constant_unknown_takes_no_region():
    d = _stacked()
    with pytest.raises((ValueError, NotImplementedError), match="region"):
        d.unknown.scalar(constant=True, region="lower")


def test_a_region_field_is_held_at_zero_outside_next_to_a_term_on_interior_faces():
    """A term on faces INSIDE the mesh (`d.tag(..., region=body)`: a heat-transfer edge between two parts)
    sends the whole form down the non-nodal assembler, which must pin the region field's outside DOFs as
    the nodal one does. Without the pins those DOFs sit in no equation and the sparse LU reports a singular
    operator.

    Same oracle as above for u (P2, -u'' = 1 on `lower`, u = y - y^2/2); T and its interior-face transfer
    term only make the route. Then the same with a complex field next to them (an eddy-current form)."""
    for complex_field in (False, True):
        d = _stacked()
        d.tag("mid", lambda x, y: np.abs(y - 1.0) < 1e-9, region="upper")
        T = d.unknown.scalar("T")
        u = d.unknown.scalar("u", order=2, region="lower")
        w, v = T.test(), u.test()
        x, y = d.variable("interior", split=True)[:2]
        xl, yl = d.variable("lower", split=True)[:2]
        xb, yb = d.variable("bottom", split=True)[:2]
        xm, ym = d.variable("mid", split=True)[:2]
        Ti, wi = T.bind(x=x, y=y), w.bind(x=x, y=y)
        ub, vb = u.bind(x=xl, y=yl), v.bind(x=xl, y=yl)
        terms = [
            Ti.d(x) * wi.d(x) + Ti.d(y) * wi.d(y) - wi,
            10.0 * (T.bind(x=xm, y=ym) - 300.0) * w.bind(x=xm, y=ym),  # h (T - T_out) on the interior face
            ub.d(xl) * vb.d(xl) + ub.d(yl) * vb.d(yl) - vb,
            T(xb, yb) - 300.0,
            u(xb, yb) - 0.0,
        ]
        if complex_field:
            A = d.unknown.scalar("A", complex=True)
            W = A.test()
            Ai, Wi = A.bind(x=x, y=y), W.bind(x=x, y=y)
            terms += [Ai.d(x) * Wi.d(x) + Ai.d(y) * Wi.d(y) + 1j * Ai * Wi - Wi, A(xb, yb) - 0.0]
        fem = jno.fem(terms)
        sol = fem.solve(linear=jno.solve.lu(backend="host"))
        uu = np.asarray(sol.u).reshape(-1)
        pts = np.asarray(d.mesh.points)[:, :2]
        n = len(pts)  # P2 numbers the mesh vertices first
        inside = pts[:, 1] <= 1.0 + 1e-12
        np.testing.assert_allclose(uu[:n][inside], (pts[:, 1] - 0.5 * pts[:, 1] ** 2)[inside], atol=1e-10)
        assert np.all(uu[:n][~inside] == 0.0), complex_field
