"""Transient remeshing honours what it is told: a ``criterion=`` steers it, and ``max_dofs`` bounds it.

``jno.solve.remesh(criterion=...)`` documents the phase-field case (``phi * (1 - phi)``) -- a transient one
by nature -- but the transient driver used to ignore the criterion and refine on the recovery estimate of
``metric_field`` instead, with no error. And its isotropic branch refined by ``refine_factor`` every remesh
while ignoring the vertex budget the docstring promised, so the mesh grew geometrically.

Oracles:
  * a criterion localised in ONE corner of a problem that is symmetric about the centre must break the
    symmetry of the mesh -- the corner it names ends up far denser than the mirrored one;
  * a field criterion on an advected hump must follow the hump's position at the time of the remesh (the
    LIVE state), not its initial one;
  * an isotropic remesh with ``max_dofs`` holds the vertex count, instead of ratcheting it up;
  * a condition criterion (``jno.le``) that already holds remeshes nothing, and ``theta`` beside a condition
    is refused, as on the steady loop.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno

pytest.importorskip("mmgpy", reason="mmgpy required for adaptive remeshing (fem_adapt imports it)")

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _heat(mesh_size=0.1, t_end=0.2, nt=11):
    """u_t = 0.1 Δu, mode-(1,1) IC, u = 0 on the boundary: symmetric about the centre of the unit square."""
    d = jno.shape.rect(0, 0, 1, 1, size=mesh_size).domain(time=(0.0, t_end, nt))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + 0.1 * (ui.x * vi.x + ui.y * vi.y),
            u(xb, yb) - 0.0,
            u(x0, y0) - jno.fn(lambda x, y: jnp.sin(PI * x) * jnp.sin(PI * y), [x0, y0]),
        ]
    )
    return fem, d, u, xi, yi


def _vertices_near(points, centre, radius):
    return int(np.sum(np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]) < radius))


def _fine_cell_centroid(points, cells, quantile=0.15):
    """Centroid of the smallest cells -- where the remesher put its resolution."""
    P = points[cells][:, :, :2]
    area = 0.5 * np.abs(
        (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 2, 0] - P[:, 0, 0]) * (P[:, 1, 1] - P[:, 0, 1])
    )
    fine = area <= np.quantile(area, quantile)
    return P[fine].mean(axis=(0, 1))


def test_fem_eval_assembles_on_a_transient_problem():
    """The criterion is assembled through `fem.eval`, which used to refuse every transient problem (the
    free-residual factory was published only after the transient branch had returned). Oracle: the P1
    basis is a partition of unity, so the load vector of the weak term `1·v` sums to the domain's area."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.2).domain(time=(0.0, 0.1, 3))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(x0, y0) - 1.0])
    load = np.asarray(fem.eval(1.0 * vi, np.zeros(fem.dofs)))
    assert abs(load.sum() - 1.0) < 1e-12, f"the load vector of 1·v sums to {load.sum():.15f}; the area is 1"


def test_a_transient_criterion_steers_the_remesh():
    """The criterion names the top-right corner of a problem symmetric about the centre. If it is read,
    the remeshed mesh is dense there and coarse in the mirrored corner; if it is ignored the recovery
    estimate of the (symmetric) solution refines both corners alike.

    Measured: ignored, the two corners held 142 and 179 vertices (and the mesh ratcheted to 2925); read,
    46 against 4 on a 186-vertex mesh. Three remeshes, so the contrast is not left to a single round."""
    fem, d, u, xi, yi = _heat()
    corner = jno.np.exp(-((xi - 0.85) ** 2 + (yi - 0.85) ** 2) / 0.005)
    traj = fem.solve(adapt=jno.solve.remesh(criterion=corner, every=3))
    pts = np.asarray(traj.meshes[-1][0])[:, :2]
    named, mirror = _vertices_near(pts, (0.85, 0.85), 0.12), _vertices_near(pts, (0.15, 0.15), 0.12)
    assert named > 3 * max(mirror, 1), (
        f"the criterion did not steer the remesh: {named} vertices at its corner vs {mirror} mirrored"
    )


def test_a_field_criterion_reads_the_live_state():
    """A hump advected from x = 0.25 at unit speed: a criterion on the FIELD must refine where the hump is
    when the remesh happens, which is only possible if the criterion is evaluated on the current state."""
    nt, t_end, every = 21, 0.5, 5
    d = jno.shape.rect(0, 0, 1, 1, size=0.06).domain(time=(0.0, t_end, nt))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    x0, y0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    hump = jno.fn(lambda x, y: jnp.exp(-((x - 0.25) ** 2 + (y - 0.5) ** 2) / 0.01), [x0, y0])
    fem = jno.fem([ui.t * vi + ui.x * vi + 0.01 * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(x0, y0) - hump])
    uc = u.bind(x=xi, y=yi)
    traj = fem.solve(adapt=jno.solve.remesh(criterion=uc * uc, every=every))
    dt = t_end / (nt - 1)
    last_remesh_t = dt * every * ((nt - 2) // every)  # the final frame's mesh was built here
    pts, cells = (np.asarray(a) for a in traj.meshes[-1])
    cx, cy = _fine_cell_centroid(pts, cells)
    assert abs(cx - (0.25 + last_remesh_t)) < 0.1 and abs(cy - 0.5) < 0.1, (
        f"the refinement sits at ({cx:.2f}, {cy:.2f}); the hump was at ({0.25 + last_remesh_t:.2f}, 0.50) when the mesh was built"
    )


def test_an_isotropic_remesh_holds_the_budget():
    """Isotropic marking refines the marked cells by refine_factor; on a march it must also coarsen the
    rest so the vertex count stays near the budget, instead of growing by ~refine_factor^dim per remesh."""
    fem, *_ = _heat(mesh_size=0.1, t_end=0.24, nt=16)
    budget = 1500
    traj = fem.solve(adapt=jno.solve.remesh(every=3, max_dofs=budget))
    post = [m[0].shape[0] for m in traj.meshes][4:]  # after the first remesh
    assert max(post) < 1.5 * min(post), f"the vertex count ratcheted instead of holding the budget: {post}"
    assert 0.4 * budget < np.median(post) < 2.5 * budget, (
        f"the budget was {budget} vertices; the mesh held {np.median(post):.0f}"
    )


def test_a_condition_that_holds_remeshes_nothing():
    """A condition criterion is a trigger on a march: the mesh is rebuilt only when some cell breaks it."""
    fem, *_ = _heat(mesh_size=0.1, t_end=0.2, nt=11)
    n0 = int(np.asarray(fem.domain.mesh.points).shape[0])
    traj = fem.solve(adapt=jno.solve.remesh(criterion=lambda d: jno.le(d.cell_aspect(), 50.0), every=2))
    assert all(int(m[0].shape[0]) == n0 for m in traj.meshes), "a condition that holds everywhere still remeshed"
    assert not any(h.get("remeshed") for h in fem.adapt_history), fem.adapt_history


def test_an_anisotropic_steady_remesh_reads_the_criterion():
    """The steady loop's anisotropic branch built its metric from the SOLUTION's Hessian whatever the
    criterion said. The criterion names one corner of a problem symmetric about the centre."""
    d = jno.shape.rect(0, 0, 1, 1, size=0.1).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - 1.0 * vi, u(xb, yb) - 0.0])
    corner = jno.np.exp(-((xi - 0.85) ** 2 + (yi - 0.85) ** 2) / 0.005)
    fem.solve(adapt=jno.solve.remesh(criterion=corner, anisotropic=True, max_iters=2))
    pts = np.asarray(fem.domain.mesh.points)[:, :2]
    named, mirror = _vertices_near(pts, (0.85, 0.85), 0.12), _vertices_near(pts, (0.15, 0.15), 0.12)
    assert named > 3 * max(mirror, 1), f"the anisotropic metric ignored the criterion: {named} vs {mirror} mirrored"


def test_theta_beside_a_condition_is_refused_on_a_march():
    fem, *_ = _heat(mesh_size=0.14, t_end=0.1, nt=5)
    with pytest.raises(ValueError, match="theta"):
        fem.solve(adapt=jno.solve.remesh(criterion=lambda d: jno.le(d.cell_aspect(), 2.0), every=2, theta=0.3))
