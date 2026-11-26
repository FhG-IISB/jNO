"""``jno.fvm(..., centering="vertex")``: median-dual control volumes, nodal unknowns, strong Dirichlet rows."""

import jax
import numpy as np
import pytest

pytest.importorskip("shapely")

import jno  # noqa: E402
import jno.jnp_ops as jnn  # noqa: E402
from jno._fvm.geometry import dual_graph  # noqa: E402

PI = np.pi
MESHES = {
    "line": lambda: jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.1).domain(),
    "triangle": lambda: jno.shape.rect(0, 0, 1, 1, size=0.1).domain(),
    "quad": lambda: jno.shape.rect(0, 0, 1, 1).structured(n=8).quad().domain(),
    "tetra": lambda: jno.shape.box(0, 0, 0, 1, 1, 1, size=0.3).domain(),
    "hexahedron": lambda: jno.shape.box(0, 0, 0, 1, 1, 1).structured(n=4).quad().domain(),
}


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("kind", list(MESHES))
def test_the_median_dual_is_closed_and_fills_the_domain(kind):
    d = MESHES[kind]()
    cells, k = d._cells_topo()
    g = dual_graph(np.asarray(d.mesh_connectivity["points"])[:, : d.dimension], cells, k)
    acc = np.zeros((g.n, g.dim))
    np.add.at(acc, g.owner, g.area)
    np.add.at(acc, g.neigh, -g.area)
    np.add.at(acc, g.b_owner, g.b_area)
    assert np.abs(acc).max() < 1e-14
    assert g.volume.sum() == pytest.approx(1.0, abs=1e-13)
    assert np.linalg.norm(g.b_area, axis=1).sum() == pytest.approx(2.0 * g.dim, abs=1e-12)
    assert np.all(np.einsum("fd,fd->f", g.area, g.delta) > 0)


@pytest.mark.parametrize("quad", [False, True], ids=["triangles", "quads"])
def test_vertex_poisson_is_second_order_at_the_nodes(quad):
    errs = []
    for h in (0.1, 0.05):
        d = (
            jno.shape.rect(0, 0, 1, 1).structured(n=int(1 / h)).quad().domain()
            if quad
            else jno.shape.rect(0, 0, 1, 1, size=h).domain()
        )
        x, y, _ = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        u = d.unknown()
        ui = u.bind(x=x, y=y)
        p = jno.fvm([-(ui.xx + ui.yy) - 2 * PI**2 * jnn.sin(PI * x) * jnn.sin(PI * y), u(xb, yb) - 0.0], centering="vertex")
        sol = np.asarray(p.solve())
        nodes = np.asarray(d.mesh_connectivity["points"])[:, :2]
        np.testing.assert_allclose(p.centroids, nodes)  # the unknowns live at the mesh nodes
        exact = np.sin(PI * nodes[:, 0]) * np.sin(PI * nodes[:, 1])
        errs.append(np.linalg.norm(sol - exact) / np.linalg.norm(exact))
    assert errs[0] < 1.5e-2 and errs[0] / errs[1] > 3.5


def test_vertex_conditions_dirichlet_neumann_and_radiation_are_exact_for_a_linear_field():
    d = jno.shape.rect(0, 0, 1, 1, size=0.08).domain()
    x, y, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    nr = d.variable("right", normals=True)
    u = d.unknown()
    ui, ur = u.bind(x=x, y=y), u.bind(x=xr, y=yr)
    c = np.asarray(d.mesh_connectivity["points"])[:, :2]
    a = np.asarray(jno.fvm([-(ui.xx + ui.yy), u(xl, yl) - 1.0, ur.d(nr) - 2.0], centering="vertex").solve())
    assert np.abs(a - (1 + 2 * c[:, 0])).max() < 1e-10
    p = jno.fvm([-(ui.xx + ui.yy), u(xl, yl) - 1.0, ur.d(nr) - 1.0 + (ur**4 - 16.0)], centering="vertex")
    assert np.abs(np.asarray(p.solve(x0=np.ones(p._Ntot))) - (1 + c[:, 0])).max() < 1e-10


@pytest.mark.parametrize(
    "scheme", [jno.solve.theta(0.5), jno.solve.erk("ssprk3", cfl=0.8)], ids=["crank-nicolson", "ssprk3"]
)
def test_vertex_heat_decay(scheme):
    """The Dirichlet rows are algebraic (zero mass): an implicit step and erk both keep them."""
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.08).domain(time=(0.0, 0.5, 100))
    x, y, t = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi, yi, _ = d.variable("initial", split=True)
    u = d.unknown()
    ui = u.bind(x=x, y=y, t=t)
    p = jno.fvm(
        [ui.t - 0.05 * (ui.xx + ui.yy), u(xb, yb) - 0.0, u(xi, yi) - jnn.sin(PI * xi) * jnn.sin(PI * yi)],
        centering="vertex",
    )
    traj = np.asarray(p.solve(time=scheme))
    c = p.centroids
    exact = np.exp(-2 * 0.05 * PI**2 * 0.5) * np.sin(PI * c[:, 0]) * np.sin(PI * c[:, 1])
    assert np.linalg.norm(traj[-1] - exact) / np.linalg.norm(exact) < 6e-3
    boundary = np.asarray(d.mesh_connectivity["boundary_indices"])
    assert np.abs(traj[:, boundary]).max() < 1e-12


def test_vertex_convection_stays_bounded():
    d = jno.Path(0.0, 0.0).line_to(1.0, 0.0).curve(size=0.05).domain()
    x = d.variable("interior", split=True)[0]
    xl, xr = d.variable("left", split=True)[0], d.variable("right", split=True)[0]
    u = d.unknown()
    ui = u.bind(x=x)
    s = np.asarray(jno.fvm([ui.d(x, scheme=jno.fv()) - 0.01 * ui.xx, u(xl) - 0.0, u(xr) - 1.0], centering="vertex").solve())
    assert s.min() > -1e-3 and s.max() < 1 + 1e-12
