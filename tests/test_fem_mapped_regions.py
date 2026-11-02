"""Mapped regions, ``.attach(map=f)``: a region of the mesh stands for another region of space, and every weak term
written there is evaluated by the change of variables. Each case closes a finite mesh with a mapped layer whose
outer edge is infinity and compares with the exact unbounded-space solution.

  * current loop (axisymmetric A_phi), inversion shell       -> elliptic-integral field
  * sphere at T1 in an unbounded medium, nonzero far value    -> T_inf + (T1 - T_inf) a / rho
  * line source under a grounded surface, axis maps + corners -> image pair
  * 2-D net source split off, inversion on the rest           -> exact superposition
  * outgoing wave, COMPLEX radial map (absorbing layer)       -> Hankel H0(1)
  * a map that moves an interface raises
"""

import numpy as np
import pytest
from jax import config
from scipy.special import ellipe, ellipk, hankel1, j1

import jno

config.update("jax_enable_x64", True)
disk, rect = jno.shape.disk, jno.shape.rect
MU0 = 4e-7 * np.pi


def half(rad, zc=0.0):
    return disk(0, zc, rad) & rect(0, zc - rad, rad, zc + rad)


def inversion(R, centre=(0.0, 0.0)):
    """rho = R^2 / (2R - rho'): the shell R <= rho' <= 2R covers R <= rho < infinity."""
    c1, c2 = centre

    def f(x, y):
        rho = jno.np.sqrt((x - c1) ** 2 + (y - c2) ** 2)
        q = R**2 / ((2 * R - rho) * rho)
        return c1 + (x - c1) * q, c2 + (y - c2) * q

    return f


def _source(d, name, q):
    pts, tri = np.asarray(d.mesh.points)[:, :2], np.asarray(d.mesh.cells_dict["triangle"])
    blocks = [c.type == "triangle" for c in d.mesh.cells]
    ix = np.concatenate([np.asarray(i) for i, b in zip(d.mesh.cell_sets[name], blocks) if b and i is not None and len(i)])
    p = pts[tri[ix]]
    area = 0.5 * np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    )
    d.attach(name, f=q / area.sum())


def _grad_dot(u, v, x, y):
    return u.d(x) * v.d(x) + u.d(y) * v.d(y)


@pytest.mark.parametrize("R", [0.08, 0.15])
def test_current_loop_in_free_space(R):
    a, w, I = 0.05, 0.002, 1.0  # noqa: E741
    d = (
        (
            rect(a - w / 2, -w / 2, a + w / 2, w / 2).name("ring").attach(J=I / w**2).size(w / 3)
            + half(R).name("air").attach(J=0.0)
            + (half(2 * R) - half(R)).name("infinity").attach(J=0.0, map=inversion(R))
        )
        .sized(0.004)
        .domain()
    )
    A, W = d.fem_symbols(names=("A", "W"))
    r, z, _ = d.variable("interior", split=True)
    Ai, Wi = A.bind(r=r, z=z), W.bind(r=r, z=z)
    dV = 2 * np.pi * r
    fem = jno.fem(
        [
            (1 / MU0) * (Ai.d(z) * Wi.d(z) + (Ai.d(r) + Ai / r) * (Wi.d(r) + Wi / r)) * dV - d.J * Wi * dV,
            A(*d.variable("left", split=True)[:2]) - 0.0,
            A(*d.variable("arc", split=True)[:2]) - 0.0,
        ]
    )
    sol = np.asarray(fem.solve(linear=jno.solve.lu()))
    p = np.asarray(d.mesh.points)[:, :2]
    sel = (np.hypot(p[:, 0] - a, p[:, 1]) > 0.01) & (p[:, 0] > 1e-6) & (np.hypot(*p.T) < R)
    m = lambda r_, z_, a_: 4 * a_ * r_ / ((a_ + r_) ** 2 + z_**2)  # noqa: E731
    loop = lambda r_, z_, a_, z0: (
        MU0
        * I
        / (np.pi * np.sqrt(m(r_, z_ - z0, a_)))
        * np.sqrt(a_ / r_)  # noqa: E731
        * ((1 - m(r_, z_ - z0, a_) / 2) * ellipk(m(r_, z_ - z0, a_)) - ellipe(m(r_, z_ - z0, a_)))
    )
    g, gw = np.polynomial.legendre.leggauss(5)
    exact = (
        sum(
            wi * wj * loop(p[sel, 0], p[sel, 1], a + gi * w / 2, gj * w / 2)
            for gi, wi in zip(g, gw)
            for gj, wj in zip(g, gw)
        )
        / 4
    )
    assert np.abs(sol[sel] - exact).max() / np.abs(exact).max() < 2.5e-3


def test_sphere_in_an_unbounded_medium_reaches_the_far_value():
    """Axisymmetric heat from a sphere at T1; T_inf held on the shell's outer edge, which is infinity. A 1/rho field
    in an axisymmetric shell wants the shell graded towards that edge (measured: h^0.8 uniform, h^2 graded)."""
    a, R, T1, Tinf, h = 0.05, 0.15, 900.0, 300.0, 0.005

    def size(x, y, z=0.0):
        t = np.hypot(x, y) - R
        return h if t <= 0 else max(h * min(1.0, 2.0 * (R - t) / R), 0.25 * h)

    d = (
        ((half(R) - half(a)).name("medium") + (half(2 * R) - half(R)).name("infinity").attach(map=inversion(R)))
        .sized(size)
        .domain()
    )
    d.tag("hot", lambda x, y: np.abs(np.hypot(x, y) - a) < h / 4)
    d.tag("far", lambda x, y: np.abs(np.hypot(x, y) - 2 * R) < h / 4)
    T, v = d.fem_symbols(names=("T", "v"))
    r, z, _ = d.variable("interior", split=True)
    fem = jno.fem(
        [
            _grad_dot(T.bind(r=r, z=z), v.bind(r=r, z=z), r, z) * 2 * np.pi * r,
            T(*d.variable("hot", split=True)[:2]) - T1,
            T(*d.variable("far", split=True)[:2]) - Tinf,
        ]
    )
    sol = np.asarray(fem.solve(linear=jno.solve.lu()))
    p = np.asarray(d.mesh.points)[:, :2]
    rho = np.hypot(*p.T)
    inside = (rho > a * 1.001) & (rho < R)
    assert np.abs(sol[inside] - (Tinf + (T1 - Tinf) * a / rho[inside])).max() / (T1 - Tinf) < 2e-3


def _axis_map(x0, side, L, k=None):
    """Stretch one coordinate from x0 outwards: x = x0 + side * t L / (L - t), t = side (x' - x0); with k, the
    Bermudez absorbing part (i/k) * (-ln(1 - t/L)) is added."""

    def g(x):
        t = side * (x - x0)
        h = t * L / (L - t)
        if k is not None:
            h = t - 1j / k * jno.np.log(1 - t / L)
        return x0 + side * h

    return g


def _box_with_layers(x0, x1, y0, y1, L, open_sides, src, h, k=None):
    cx, cy, a0, q = src
    sides = {"x-": (x0, -1), "x+": (x1, +1), "y-": (y0, -1), "y+": (y1, +1)}
    span = {"x-": (x0 - L, x0), "x+": (x1, x1 + L), "y-": (y0 - L, y0), "y+": (y1, y1 + L)}
    plan = disk(cx, cy, a0).name("src") + rect(x0, y0, x1, y1).name("core").attach(f=0.0)
    xs = [s for s in ("x-", "x+") if s in open_sides] + [None]
    ys = [s for s in ("y-", "y+") if s in open_sides] + [None]
    for sx in xs:
        for sy in ys:
            if sx is None and sy is None:
                continue
            gx = _axis_map(*sides[sx], L, k) if sx else (lambda x: x)
            gy = _axis_map(*sides[sy], L, k) if sy else (lambda y: y)
            (ax, bx), (ay, by) = span[sx] if sx else (x0, x1), span[sy] if sy else (y0, y1)
            plan = plan + rect(ax, ay, bx, by).name("L" + (sx or "") + (sy or "")).attach(
                f=0.0, map=lambda x, y, gx=gx, gy=gy: (gx(x), gy(y))
            )
    d = plan.sized(h).domain()
    _source(d, "src", q)
    u, v = d.fem_symbols(names=("u", "v"))
    x, y, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    sol = np.asarray(
        jno.fem(
            [
                _grad_dot(ui, vi, x, y) - (k or 0.0) ** 2 * ui * vi - d.f * vi,
                u(*d.variable("boundary", split=True)[:2]) - 0.0,
            ],
            quad_degree=3,
        ).solve(linear=jno.solve.lu())
    )
    p = np.asarray(d.mesh.points)[:, :2]
    core = (
        (p[:, 0] >= x0)
        & (p[:, 0] <= x1)
        & (p[:, 1] >= y0)
        & (p[:, 1] <= y1)
        & (np.hypot(p[:, 0] - cx, p[:, 1] - cy) > 1.5 * a0)
    )
    return p[core], sol[core]


def test_line_source_under_a_grounded_surface():
    src = (0.0, -0.12, 0.02, 1.0)
    p, u = _box_with_layers(-0.3, 0.3, -0.3, 0.0, 0.3, {"x-", "x+", "y-"}, src, h=0.01)
    exact = src[3] / (2 * np.pi) * np.log(np.hypot(p[:, 0], p[:, 1] + src[1]) / np.hypot(p[:, 0], p[:, 1] - src[1]))
    assert np.abs(u - exact).max() / np.abs(exact).max() < 5e-3


def test_net_source_split_off_and_the_rest_mapped():
    srcs = [((-0.4, 0.0), 0.05, 1.0), ((0.4, 0.2), 0.05, 0.5)]
    Q, ac, R = 1.5, 0.1, 1.0
    plan = disk(-0.4, 0, 0.05).name("s0") + disk(0.4, 0.2, 0.05).name("s1") + disk(0, 0, ac).name("centre")
    plan = (
        plan
        + disk(0, 0, R).name("medium").attach(f=0.0)
        + (disk(0, 0, 2 * R) - disk(0, 0, R)).name("infinity").attach(f=0.0, map=inversion(R))
    )
    d = plan.sized(0.03).domain()
    for i, (_, _, q) in enumerate(srcs):
        _source(d, f"s{i}", q)
    _source(d, "centre", -Q)
    w, v = d.fem_symbols(names=("w", "v"))
    x, y, _ = d.variable("interior", split=True)
    sol = np.asarray(
        jno.fem(
            [
                _grad_dot(w.bind(x=x, y=y), v.bind(x=x, y=y), x, y) - d.f * v.bind(x=x, y=y),
                w(*d.variable("boundary", split=True)[:2]) - 0.0,
            ]
        ).solve(linear=jno.solve.lu())
    )
    p = np.asarray(d.mesh.points)[:, :2]
    G = lambda c: -np.log(np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])) / (2 * np.pi)  # noqa: E731
    away = (np.hypot(*p.T) < R) & (np.hypot(*p.T) > 1.5 * ac)
    for c, a0, _ in srcs:
        away &= np.hypot(p[:, 0] - c[0], p[:, 1] - c[1]) > 1.5 * a0
    exact = sum(q * G(c) for c, _, q in srcs) - Q * G((0.0, 0.0))
    assert np.abs(sol[away] - exact[away]).max() / np.abs(exact[away]).max() < 2e-2


def test_complex_map_absorbs_an_outgoing_wave():
    k, R, L, a0 = 2 * np.pi / 0.2, 0.5, 0.2, 0.03

    def absorb(x, y):  # Bermudez: rho~ = rho' + (i/k) * (-ln(1 - s/L)), s = rho' - R
        rho = jno.np.sqrt(x**2 + y**2)
        q = (rho - 1j / k * jno.np.log(1 - (rho - R) / L)) / rho
        return x * q, y * q

    d = (
        (
            disk(0, 0, a0).name("src")
            + disk(0, 0, R).name("medium").attach(f=0.0)
            + (disk(0, 0, R + L) - disk(0, 0, R)).name("pml").attach(f=0.0, map=absorb)
        )
        .sized(0.008)
        .domain()
    )
    _source(d, "src", 1.0)
    u, v = d.fem_symbols(names=("u", "v"))
    x, y, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    sol = np.asarray(
        jno.fem(
            [_grad_dot(ui, vi, x, y) - k**2 * ui * vi - d.f * vi, u(*d.variable("boundary", split=True)[:2]) - 0.0],
            quad_degree=3,
        ).solve(linear=jno.solve.lu())
    )
    p = np.asarray(d.mesh.points)[:, :2]
    rho = np.hypot(*p.T)
    keep = (rho > 2 * a0) & (rho < R)
    exact = 2 * j1(k * a0) / (k * a0) * 0.25j * hankel1(0, k * rho[keep])
    assert np.iscomplexobj(sol)
    assert np.abs(sol[keep] - exact).max() / np.abs(exact).max() < 5e-2


def test_a_map_that_moves_an_interface_raises():
    d = (
        (
            disk(0, 0, 0.5).name("medium")
            + (disk(0, 0, 1.0) - disk(0, 0, 0.5)).name("shell").attach(map=lambda x, y: (2.0 * x, 2.0 * y))
        )
        .sized(0.1)
        .domain()
    )
    u, v = d.fem_symbols(names=("u", "v"))
    x, y, _ = d.variable("interior", split=True)
    with pytest.raises(ValueError, match="not the identity"):
        jno.fem([_grad_dot(u.bind(x=x, y=y), v.bind(x=x, y=y), x, y)])


def test_a_second_build_with_the_same_coordinates_maps_the_same():
    """jno.fem retags a term's coordinate Variables in place ("interior" -> "fem_gauss"); a second form built from
    the same Variables must still be mapped (it once silently was not: the shell then acted as truncated air)."""
    R = 0.1
    d = (
        (
            disk(0, 0, 0.02).name("src").attach(f=1.0)
            + disk(0, 0, R).name("medium").attach(f=0.0)
            + (disk(0, 0, 2 * R) - disk(0, 0, R)).name("infinity").attach(f=0.0, map=inversion(R))
        )
        .sized(0.01)
        .domain()
    )
    w, v = d.fem_symbols(names=("w", "v"))
    x, y, _ = d.variable("interior", split=True)
    wi, vi = w.bind(x=x, y=y), v.bind(x=x, y=y)
    bc = w(*d.variable("boundary", split=True)[:2]) - 0.0
    first = np.asarray(jno.fem([_grad_dot(wi, vi, x, y) + wi * vi - d.f * vi, bc]).solve(linear=jno.solve.lu()))
    second = np.asarray(jno.fem([_grad_dot(wi, vi, x, y) + wi * vi - d.f * vi, bc]).solve(linear=jno.solve.lu()))
    np.testing.assert_allclose(second, first, rtol=1e-12, atol=1e-14)
