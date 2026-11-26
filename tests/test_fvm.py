"""``jno.fvm``: the finite-volume solve of a strong-form constraint list (steady diffusion and reaction)."""

import jax
import numpy as np
import pytest

pytest.importorskip("shapely")

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _rel(a, b):
    return float(np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b))


def _square(h, quad=False):
    if quad:
        return jno.shape.rect(0, 0, 1, 1).structured(n=int(round(1 / h))).quad().domain()
    return jno.shape.rect(0, 0, 1, 1, size=h).domain()


def _poisson(d):
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    f = 2 * PI**2 * jnn.sin(PI * x) * jnn.sin(PI * y)
    p = jno.fvm([-(ui.xx + ui.yy) - f, u(xb, yb) - 0.0])
    c = p.centroids
    return _rel(p.solve(), np.sin(PI * c[:, 0]) * np.sin(PI * c[:, 1]))


@pytest.mark.parametrize("quad", [False, True], ids=["triangles", "quads"])
def test_poisson_is_second_order(quad):
    errs = [_poisson(_square(h, quad)) for h in (0.1, 0.05)]
    assert errs[0] < 1e-2
    assert errs[0] / errs[1] > 3.5  # h^2: a factor 4 per halving


def test_returns_one_value_per_cell_at_the_centroids():
    d = _square(0.2)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    p = jno.fvm([-(ui.xx + ui.yy) - 1.0, u(xb, yb) - 0.0])
    sol = p.solve()
    cells, _ = d._cells_topo()
    assert sol.shape == (cells.shape[0],) and p.centroids.shape == (cells.shape[0], 2)
    assert p.volumes.sum() == pytest.approx(1.0)
    assert p.to_nodes(sol).shape == (np.asarray(d.mesh_connectivity["points"]).shape[0],)


@pytest.mark.parametrize("quad", [False, True], ids=["triangles", "quads"])
def test_a_linear_field_is_exact_with_every_kind_of_condition(quad):
    """u = 1 + 2x: Dirichlet on the left, Neumann on the right, natural (zero flux) top and bottom. The
    least-squares gradients are exact for a linear field, so the solve is, on a skewed mesh too."""
    d = _square(0.08, quad)
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui, ur = u.bind(x=x, y=y), u.bind(x=xr, y=yr)
    p = jno.fvm([-(ui.xx + ui.yy), u(xl, yl) - 1.0, ur.d(nr) - 2.0])
    c = p.centroids
    assert np.abs(np.asarray(p.solve()) - (1 + 2 * c[:, 0])).max() < 1e-8


def test_a_nonlinear_radiation_condition():
    """``∂u/∂n = 1 - (u⁴ - 16)`` on the right edge is met by u = 1 + x (u = 2, ∂u/∂n = 1 there)."""
    d = _square(0.08)
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui, ur = u.bind(x=x, y=y), u.bind(x=xr, y=yr)
    p = jno.fvm([-(ui.xx + ui.yy), u(xl, yl) - 1.0, ur.d(nr) - 1.0 + (ur**4 - 16.0)])
    c = p.centroids
    assert np.abs(np.asarray(p.solve(x0=np.ones(p._Ntot))) - (1 + c[:, 0])).max() < 1e-8


def test_mixed_dirichlet_neumann_robin():
    """The fdm tutorial's u = y²: bottom u = 0, left/right insulated, top ∂u/∂n + (u - 3) = 0."""
    d = _square(0.05)
    x, y, _ = d.variable("interior", split=True)
    xbo, ybo, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    nt = d.variable("top", normals=True)
    u = d.unknown()
    ui, ut = u.bind(x=x, y=y), u.bind(x=xt, y=yt)
    p = jno.fvm([-ui.xx - ui.yy + 2.0, u(xbo, ybo) - 0.0, ut.d(nt) + (ut - 3.0)])
    assert _rel(p.solve(), p.centroids[:, 1] ** 2) < 2e-3  # cell averages of y² vs its centroid values


def test_a_solution_dependent_diffusivity_is_second_order():
    """-∇·((1 + u)∇u) = f, manufactured from u = sin(πx)sin(πy)."""
    errs = []
    for h in (0.1, 0.05):
        d = _square(h)
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        u = d.unknown()
        ui = u.bind(x=x, y=y)
        ue = jnn.sin(PI * x) * jnn.sin(PI * y)
        grad2 = PI**2 * (jnn.cos(PI * x) ** 2 * jnn.sin(PI * y) ** 2 + jnn.sin(PI * x) ** 2 * jnn.cos(PI * y) ** 2)
        f = (1 + ue) * 2 * PI**2 * ue - grad2
        k = 1 + ui
        p = jno.fvm([-(k * ui.x).x - (k * ui.y).y - f, u(xb, yb) - 0.0])
        c = p.centroids
        errs.append(_rel(p.solve(), np.sin(PI * c[:, 0]) * np.sin(PI * c[:, 1])))
    assert errs[0] < 1e-2 and errs[0] / errs[1] > 3.5


def test_fluxes_are_conserved_to_round_off():
    """-Δu = 1 in the unit square, u = 0 on the boundary: the face fluxes telescope, so the outflow through
    the boundary equals the total source exactly, at any resolution."""
    d = _square(0.07)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    sol = jax.numpy.asarray(jno.fvm([-(ui.xx + ui.yy) - 1.0, u(xb, yb) - 0.0]).solve())
    flux = jno.fvm([-(ui.xx + ui.yy), u(xb, yb) - 0.0])._residual_fn()(sol)
    assert float(flux.sum()) == pytest.approx(1.0, abs=1e-12)


def test_one_dimensional():
    errs = []
    for h in (0.05, 0.025):
        d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=h).domain()
        x = d.variable("interior", split=True)[0]
        xb = d.variable("boundary", split=True)[0]
        u = d.unknown()
        ui = u.bind(x=x)
        p = jno.fvm([-ui.xx - PI**2 * jnn.sin(PI * x), u(xb) - 0.0])
        errs.append(_rel(p.solve(), np.sin(PI * p.centroids[:, 0])))
    assert errs[0] < 5e-3 and errs[0] / errs[1] > 3.5


@pytest.mark.parametrize("hex_", [False, True], ids=["tets", "hexes"])
def test_three_dimensional_poisson(hex_):
    d = (
        jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=6).quad().domain()
        if hex_
        else jno.shape.box(0, 0, 0, 1, 1, 1, size=0.2).domain()
    )
    x, y, z = d.variable("interior", split=True)[:3]
    xb, yb, zb = d.variable("boundary", split=True)[:3]
    u = d.unknown()
    ui = u.bind(x=x, y=y, z=z)
    f = 3 * PI**2 * jnn.sin(PI * x) * jnn.sin(PI * y) * jnn.sin(PI * z)
    p = jno.fvm([-(ui.xx + ui.yy + ui.zz) - f, u(xb, yb, zb) - 0.0])
    assert _rel(p.solve(), np.prod(np.sin(PI * p.centroids), axis=1)) < 4e-2


def test_a_nodal_data_coefficient():
    """κ given as a data field at the mesh nodes (a ``jno.np.parameter`` without an optimizer): averaged to
    the faces, and the same answer as the formula."""
    d = _square(0.08)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    pts = np.asarray(d.mesh_connectivity["points"])[:, :2]
    kappa = jno.np.parameter((pts.shape[0],), name="kappa")
    import equinox as eqx

    kappa.model.module = eqx.tree_at(lambda m: m.value, kappa.model.module, jax.numpy.asarray(2.0 + 0 * pts[:, 0]))
    a = jno.fvm([-(kappa * ui.x).x - (kappa * ui.y).y - 2.0, u(xb, yb) - 0.0]).solve()
    b = jno.fvm([-(ui.xx + ui.yy) - 1.0, u(xb, yb) - 0.0]).solve()
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_a_non_conservative_term_warns():
    d = _square(0.2)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    with pytest.warns(UserWarning, match="not conservative"):
        jno.fvm([-(ui.xx + ui.yy) + ui * ui.x - 1.0, u(xb, yb) - 0.0])


def test_two_conditions_on_the_same_faces_raise():
    d = _square(0.2)
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xl, yl, _ = d.variable("left", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y)
    with pytest.raises(ValueError, match="two conditions"):
        jno.fvm([-(ui.xx + ui.yy) - 1.0, u(xb, yb) - 0.0, u(xl, yl) - 1.0])
