"""A moving mesh is remeshed when it degrades: ``adapt=jno.solve.remesh(criterion=<mesh-geometry condition>)``.

A geometry term moves the vertices at fixed connectivity, so a large motion degrades the elements until the
march stops being accurate (or tangles). With a condition on the element shape, the driver marches in
chunks of ``every`` steps, checks the condition on the moved mesh, and -- only where it breaks -- remeshes,
rebuilds the problem on the new mesh and carries the state across.

Oracles:

* a condition that always holds never remeshes, and the march is exactly the plain march;
* a top edge bulging as ``y' = k y sin(pi x)`` degrades the mesh (maximum cell aspect 1.32 -> 6.27 by
  T = 0.5 with no remesh, measured). Under ``aspect <= 3`` the march completes on a better mesh, the
  surface ends where the plain march puts it, and a constant field is still exactly constant after every
  transfer.
"""

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


def _aspect(P, C):
    """Max edge x perimeter / (4 sqrt(3) area): 1 for an equilateral triangle, larger when stretched."""
    t = np.asarray(P)[np.asarray(C)]
    e = np.stack([np.linalg.norm(t[:, i] - t[:, j], axis=1) for i, j in ((1, 2), (2, 0), (0, 1))], 1)
    p, q = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    a = 0.5 * np.abs(p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0])
    return e.max(1) * e.sum(1) / (4.0 * np.sqrt(3.0) * a)


def _bulge(adapt=None, *, k=2.0, T=0.5, n=26, region="boundary"):
    d = jno.shape.rect(0.0, 0.0, 1.0, 1.0, size=0.1).domain(time=(0.0, T, n))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, tb = d.variable(region, split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + 0.1 * (ui.x * vi.x + ui.y * vi.y),
            u(ci[0], ci[1]) - 1.0,
            yb.d(tb) - k * yb * jno.np.sin(np.pi * xb),  # the top bulges; the other edges hold (sin = 0, y = 0)
        ]
    )
    return fem, (fem.solve() if adapt is None else fem.solve(adapt=adapt))


def test_a_condition_that_always_holds_changes_nothing():
    _, plain = _bulge(T=0.2, n=7)
    fem, held = _bulge(jno.solve.remesh(criterion=lambda d: jno.le(d.cell_aspect(), 100.0), every=2), T=0.2, n=7)
    assert len(held.states) == len(plain.states)
    for k in range(len(plain.states)):
        assert np.allclose(np.asarray(held.meshes[k][0]), np.asarray(plain.meshes[k][0]), atol=1e-12), f"mesh {k}"
        assert np.allclose(np.asarray(held.states[k]), np.asarray(plain.states[k]), atol=1e-12), f"state {k}"
    assert fem.adapt_history and not any(h["remeshed"] for h in fem.adapt_history)


def test_a_degrading_mesh_is_remeshed_and_marches_on():
    _, plain = _bulge()
    fem, run = _bulge(jno.solve.remesh(criterion=lambda d: jno.le(d.cell_aspect(), 3.0), every=1))
    assert any(h["remeshed"] for h in fem.adapt_history), "the condition broke and nothing was remeshed"
    worst_plain = _aspect(*plain.meshes[-1]).max()
    worst = _aspect(*run.meshes[-1]).max()
    assert worst_plain > 6.0, "the oracle did not degrade, so the comparison proves nothing"
    assert worst < 0.6 * worst_plain, f"remeshed march ends at aspect {worst:.2f} (plain {worst_plain:.2f})"
    y_plain = np.asarray(plain.meshes[-1][0])[:, 1].max()
    y_run = np.asarray(run.meshes[-1][0])[:, 1].max()
    assert abs(y_run - y_plain) < 2e-2 * y_plain, f"the surface ended at {y_run:.4f}, plain {y_plain:.4f}"
    for s in run.states:
        assert np.ptp(np.asarray(s)) < 1e-10, "a constant field did not survive a transfer"


def test_a_region_that_cannot_follow_a_remesh_is_refused():
    with pytest.raises(NotImplementedError, match="re-derive"):
        _bulge(jno.solve.remesh(criterion=lambda d: jno.le(d.cell_aspect(), 3.0)), region="top", T=0.1, n=3)


def test_a_ranking_criterion_is_refused_on_a_moving_mesh():
    """A plain expression ranks cells but cannot say when the mesh is bad enough to rebuild."""
    with pytest.raises(NotImplementedError, match="mesh-geometry CONDITION"):
        _bulge(jno.solve.remesh(criterion=lambda d: d.cell_aspect()), T=0.1, n=3)
