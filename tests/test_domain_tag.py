"""``domain.tag(name, where)`` -- name an abstract region from a spatial predicate.

One general method (no interior/boundary flag): the predicate is registered as a FEM boundary
location-function (the assembler restricts it to the boundary, so it selects the right subset for
mixed / natural BCs on a complex geometry) and as a sampling region (the PINN sampler draws the
points satisfying ``where`` each step). Spatial coordinates only, so a region is the same at every
time level of a time-dependent domain.

FEM tests need x64 (assembly runs in float64).
"""

import pytest

pytest.importorskip("shapely", reason="shapely required for the CSG domains")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

import jno  # noqa: E402

dense = lambda A: jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_tag_fem_mixed_dirichlet_and_natural_on_csg_domain():
    """On a plate with a hole (single ``boundary`` tag in CSG), ``tag`` carves boundary subsets:
    Dirichlet u=1 on the left edge, u=0 on the right, and **natural** (do-nothing zero-flux) on the
    top/bottom walls AND the hole. The solve is non-singular and the field is the expected
    left-to-right gradient flowing around the insulated hole -- impossible without sub-boundary tags."""
    L, H, r = 2.0, 1.0, 0.22
    d = jno.domain(box(0, 0, L, H).difference(Point(L / 2, H / 2).buffer(r))).build_mesh(0.07)
    d.tag("hot", lambda x, y: x < 1e-6)
    d.tag("cold", lambda x, y: x > L - 1e-6)
    # walls (y=0, H) and the hole are left untagged -> natural zero-flux
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xh, yh, _ = d.variable("hot", split=True)
    xc, yc, _ = d.variable("cold", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xh, yh) - 1.0, u(xc, yc) - 0.0])
    A = dense(fem.A)
    assert int((np.abs(A).sum(1) == 0).sum()) == 0, "tagged mixed/natural BC system must be non-singular"
    uh = np.linalg.solve(np.asarray(A), np.asarray(fem.b).reshape(-1))
    assert np.all(np.isfinite(uh)) and -1e-6 <= uh.min() and uh.max() <= 1.0 + 1e-6
    pts = np.asarray(fem.points)
    assert abs(uh[pts[:, 0] < 1e-6].mean() - 1.0) < 1e-6  # Dirichlet honoured on the 'hot' tag
    assert abs(uh[pts[:, 0] > L - 1e-6].mean() - 0.0) < 1e-6  # ... and on 'cold'
    left = uh[np.abs(pts[:, 0] - 0.5) < 0.08].mean()
    right = uh[np.abs(pts[:, 0] - 1.5) < 0.08].mean()
    assert left > right > 0.0, "natural walls/hole -> smooth decreasing gradient (insulated obstacle)"


def test_tag_location_fn_selects_only_the_predicate_boundary():
    """The registered loc-fn (consumed by the assembler) selects exactly the boundary nodes satisfying
    the spatial predicate -- the rest of the boundary is left for other BCs / natural."""
    d = jno.domain(box(0, 0, 2, 1).difference(Point(1, 0.5).buffer(0.2))).build_mesh(0.1)
    d.tag("inlet", lambda x, y: x < 1e-6)
    loc = d._make_tag_location_fn("inlet")
    bnd = np.asarray(d._mesh_pool["boundary"])
    sel = np.asarray(jax.vmap(loc)(jnp.asarray(bnd))).astype(bool)
    assert sel.any() and bool((bnd[sel, 0] < 1e-6).all()) and not bool(sel.all())


def test_tag_mesh_free_sampling_resamples_in_region():
    """Without a mesh, ``tag`` carries the abstract region: ``variable`` samples points satisfying
    the predicate, and successive ``sample`` calls draw fresh points (per-step resampling for PINNs)."""
    d = jno.domain(box(0, 0, 2, 1))  # no build_mesh
    d.tag("hot", lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.04)
    xh, yh, _ = d.variable("hot", sample=(128, None), split=True)
    pts = np.asarray(d.context["hot"]).reshape(-1, 2)
    assert pts.shape[0] == 128
    assert bool((((pts[:, 0] - 0.5) ** 2 + (pts[:, 1] - 0.5) ** 2) < 0.04 + 1e-9).all())
    a, _, _ = d.sample({"hot": (64, None)})
    b, _, _ = d.sample({"hot": (64, None)})
    assert not np.array_equal(np.asarray(a), np.asarray(b)), "region must be resampled each step"


def test_tag_dirichlet_is_boundary_restricted_and_enables_natural_outflow():
    """A tagged Dirichlet must constrain ONLY the boundary, even when the predicate selects a *thick*
    region (a band around a cylinder, not just its surface) -- otherwise the assembler (which applies a
    location-fn to every node) would pin the interior velocity and silently zero the interior
    pressure rows. Here: Stokes past a cylinder, inlet + no-slip walls/cylinder via tags, **untagged
    outlet** = natural outflow. Assert the system is non-singular (the bug gave many zero rows), the
    cylinder is no-slip, and the flow leaves through the open outlet."""
    inner, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    L, H, mu, cx, cy, r = 3.0, 1.0, 1.0, 1.0, 0.5, 0.2
    cyl = Point(cx, cy).buffer(r)
    ring = Point(cx, cy).buffer(0.45).difference(cyl).intersection(box(0, 0, L, H))
    dom = jno.domain({"bulk": box(0, 0, L, H).difference(cyl).difference(ring), "ring": ring}).build_mesh(
        0.15, sizes={"ring": 0.08}
    )
    dom.point_region("ppin", (L - 0.02, 0.5))
    dom.tag("inlet", lambda x, y: x < 1e-6)
    dom.tag("walls", lambda x, y: (y < 1e-6) | (y > H - 1e-6))
    dom.tag("cyl", lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 < (r + 0.05) ** 2)  # a thick BAND, not the surface
    u, v = dom.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = dom.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = dom.variable("interior", split=True)
    xin, yin, _ = dom.variable("inlet", split=True)
    xw, yw, _ = dom.variable("walls", split=True)
    xc, yc, _ = dom.variable("cyl", split=True)
    xpn, ypn, _ = dom.variable("ppin", split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp = p.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            mu * inner(gu, gv, n_contract=2) - pp * trace(gv),
            -q.bind(x=xi, y=yi) * trace(gu),
            u(xin, yin)[0] - 4.0 * yin * (H - yin),
            u(xin, yin)[1] - 0.0,
            u(xw, yw) - (0.0, 0.0),
            u(xc, yc) - (0.0, 0.0),
            p(xpn, ypn) - 0.0,
        ]
    )
    A = np.asarray(dense(fem.A))
    assert int((np.abs(A).sum(1) == 0).sum()) == 0, "thick boundary predicate must not over-constrain the interior"
    sol = np.linalg.solve(A, np.asarray(fem.b).reshape(-1))
    assert np.all(np.isfinite(sol))
    off = fem.offsets
    uu = sol[off[0] : off[1]].reshape(-1, 2)
    pts = np.asarray(fem.field_points[0])
    dist = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    # no-slip on the exact cylinder boundary (the velocity nodes the 'cyl' tag actually constrains)
    on_cyl = np.asarray(jax.vmap(dom._make_tag_location_fn("cyl"))(jnp.asarray(pts))).astype(bool)
    assert on_cyl.any() and float(np.max(np.abs(uu[on_cyl]))) < 1e-8, "no-slip on the cylinder surface"
    # the fix: interior fluid just OUTSIDE the cylinder carries flow -- the bug pinned it to zero
    band = (dist > r + 0.02) & (dist < r + 0.05)
    assert band.any() and float(np.max(np.abs(uu[band]))) > 0.05, (
        "interior near the cylinder must carry flow, not be pinned"
    )
    assert float(uu[pts[:, 0] > L - 0.08, 0].mean()) > 0.1, "flow leaves through the natural (untagged) outlet"


def test_tag_is_spatial_only_on_time_dependent_domain():
    """A region is purely spatial: the same predicate region is carried at every time level of a
    time-dependent domain (``where`` never receives time)."""
    n_time = 5
    d = jno.domain(box(0, 0, 2, 1), time=(0.0, 1.0, n_time))
    d.tag("hot", lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.04)
    xh, yh, th = d.variable("hot", sample=(64, None), split=True)
    arr = np.asarray(d.context["hot"])
    assert arr.shape[1] == n_time  # (batch, n_time, n, dim)
    sp = arr.reshape(-1, arr.shape[-1])
    assert bool((((sp[:, 0] - 0.5) ** 2 + (sp[:, 1] - 0.5) ** 2) < 0.04 + 1e-9).all())


def test_tag_boundary_subset_has_outward_normals():
    """A pure-boundary ``tag`` (every selected node on the boundary, e.g. x<1e-6) is promoted to a
    normals-bearing boundary tag, so ``variable(name, normals=True)`` works -- the H(div)/H(curl) flux
    and tangential BCs need it. The normals come from the polygon geometry (oriented outward via the
    interior), so they equal the analytic edge normals. (Discriminating: a sign/orientation flip fails.)"""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.34)
    truth = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, 1.0), "bottom": (0.0, -1.0)}
    preds = {
        "left": lambda x, y: x < 1e-6,
        "right": lambda x, y: x > 1 - 1e-6,
        "top": lambda x, y: y > 1 - 1e-6,
        "bottom": lambda x, y: y < 1e-6,
    }
    for name, pred in preds.items():
        d.tag("my" + name, pred)
        d.variable("my" + name, normals=True, split=True)
        n = np.asarray(d.context["n_my" + name]).reshape(-1, 2)
        assert np.allclose(np.linalg.norm(n, axis=1), 1.0)  # unit length
        np.testing.assert_allclose(n, np.broadcast_to(truth[name], n.shape), atol=1e-9)  # correct & outward


def test_tag_interior_region_keeps_no_normals():
    """A 2-D region tag (selects interior nodes, not just boundary) must NOT be promoted to a boundary
    tag -- it has no meaningful outward normal, and promoting it would break PINN interior sampling."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.3)
    d.tag("blob", lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.2**2)  # interior disk, off the boundary
    with pytest.raises(ValueError, match="no outward normals"):
        d.variable("blob", normals=True, split=True)
