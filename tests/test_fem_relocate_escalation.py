"""``relocate(escalate=)``: move the nodes first, add nodes only when moving them was not enough.

r-adaptivity is ~0.4 ms per round; a node-set change is 8-15 s, because array shapes move and the
march recompiles. So relocation should get first refusal, and h-refinement should run only when the
relocated mesh still cannot represent the solution. ``escalate=tol`` measures exactly that, rather
than assuming it: after each relocation the P1 interpolation error is taken on the mesh relocation
just produced, and the budget grows only if it is still over ``tol``.

**Why the trigger is the interpolation error and not a shape-quality floor.** The mesh r-adaptivity
produces is anisotropic *on purpose* -- elements stretched ALONG the feature resolve it far more
cheaply than elements shrunk in every direction. An isotropic measure such as ``4 sqrt(3) A/sum l^2``
calls every stretched cell bad by construction, so on this class of mesh it is anti-correlated with
the mesh being good: measured on a 600 W melt ball at 26 / 74 / 163 nodes, the interpolation error's
p90 falls 0.306 -> 0.172 -> 0.078 while the count of cells that floor rejects RISES 1 -> 5 -> 8.
Escalating on it would refine a mesh that is already right.

The indicator pairs each curvature with the mesh extent **in its own direction**
(``max_e e^T |H| e`` over the cell's edges; Alauzet & Loseille, *J. Comput. Phys.* **229** (2010)
§2.2). Multiplying the largest curvature by the longest edge regardless of direction reports the
opposite and penalises precisely the elements that are doing their job.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.fem_adapt import _interp_error_indicator


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _min_iso_quality(pts, cells):
    """``4 sqrt(3) A / sum l^2`` -- 1 for an equilateral triangle, 0 for a degenerate one."""
    a, b, c = pts[cells[:, 0]], pts[cells[:, 1]], pts[cells[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    l2 = ((b - a) ** 2).sum(1) + ((c - b) ** 2).sum(1) + ((a - c) ** 2).sum(1)
    return float((4.0 * np.sqrt(3.0) * area / np.maximum(l2, 1e-300)).min())


def _rect(nx, ny):
    """A structured triangulation of the unit square with independent x and y counts."""
    X, Y = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
    pts = np.column_stack([X.ravel(), Y.ravel()])
    cells = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            a = i * nx + j
            cells += [[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]]
    return pts, np.asarray(cells, dtype=np.int64)


def _grid(n):
    g = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(g, g)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    cells = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            cells += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    return pts, np.asarray(cells, dtype=np.int64)


def test_the_indicator_matches_the_analytic_interpolation_error():
    """``u = x^2`` has ``|H| = diag(2,0)``, so an x-edge of length h carries exactly ``2h^2``."""
    pts, cells = _grid(9)
    u = pts[:, 0] ** 2
    h = 1.0 / 8.0
    got = _interp_error_indicator(pts, cells, u, 2)
    want = 2.0 * h * h / (u.max() - u.min())
    # not exact to machine precision: the Hessian is RECOVERED from the P1 gradients by patch
    # averaging, which is one-sided on the boundary patches, so the recovered curvature there carries
    # an O(h) bias. 1e-3 is far inside what the trigger needs and far outside that bias.
    assert np.allclose(got.max(), want, rtol=1e-3), f"{got.max():.6e} vs analytic {want:.6e}"

    # second order in h: halving the mesh must quarter the error
    pts2, cells2 = _grid(17)
    got2 = _interp_error_indicator(pts2, cells2, pts2[:, 0] ** 2, 2)
    assert np.allclose(got.max() / got2.max(), 4.0, rtol=1e-3), f"ratio {got.max() / got2.max():.4f}, want 4"


def test_a_linear_field_is_represented_exactly():
    """P1 reproduces a linear field, so the indicator must vanish -- however stretched the cells are."""
    pts, cells = _grid(9)
    pts = pts * np.array([1.0, 0.02])  # squash y by 50x: every cell is a sliver by an isotropic measure
    err = _interp_error_indicator(pts, cells, 3.0 * pts[:, 0] + 2.0 * pts[:, 1], 2)
    assert err.max() < 1e-12, f"linear field gave {err.max():.3e} on stretched cells"


def test_stretching_along_the_feature_is_cheap_where_an_isotropic_measure_says_it_is_not():
    """The discriminating property, and the reason the trigger is not a quality floor.

    ``u = y^2`` curves in y only. At (near) equal node count, a mesh that is COARSE in x and FINE in y
    resolves it better than a uniform one -- it spends its nodes where the curvature is. An isotropic
    quality measure reports the opposite, because it only sees that the cells are stretched.
    """
    iso_p, iso_c = _grid(9)  # 81 nodes, uniform
    an_p, an_c = _rect(5, 17)  # 85 nodes, 4x coarser in x and 2x finer in y

    e_iso = _interp_error_indicator(iso_p, iso_c, iso_p[:, 1] ** 2, 2).max()
    e_an = _interp_error_indicator(an_p, an_c, an_p[:, 1] ** 2, 2).max()
    assert e_an < 0.5 * e_iso, f"anisotropic mesh did not resolve the feature better: {e_an:.3e} vs {e_iso:.3e}"

    # ...while the isotropic quality measure prefers the WORSE mesh, which is the whole point
    assert _min_iso_quality(an_p, an_c) < _min_iso_quality(iso_p, iso_c), (
        "this test assumes the stretched mesh scores worse on 4*sqrt(3)*A/sum(l^2); it did not"
    )


def _melt(adapt, *, n=41, T=0.06):
    """A disk with a hot spot driven into one side -- a directional feature on a moving mesh."""
    d = jno.shape.disk(0.0, 0.0, 0.5, size=0.12).domain(time=(0.0, T, n))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem(
        [
            ui.t * vi + 0.004 * (ui.x * vi.x + ui.y * vi.y) - 60.0 * jno.np.exp(-60.0 * (xi**2 + yi**2)) * vi,
            u(ci[0], ci[1]) - 0.0,
            xi.d(ti) + 0.25 * jno.np.tanh(4.0 * xi),
        ]
    )
    return fem, fem.solve(adapt=adapt)


def test_escalation_adds_nodes_only_when_the_tolerance_is_exceeded():
    """The same march, two tolerances: a loose one never escalates, a tight one does."""
    fem_hi, hi = _melt(jno.solve.relocate(method="monge_ampere", every=5, relax=10, escalate=50.0))
    fem_lo, lo = _melt(jno.solve.relocate(method="monge_ampere", every=5, relax=10, escalate=0.02))

    h_hi = getattr(fem_hi, "adapt_history", []) or []
    h_lo = getattr(fem_lo, "adapt_history", []) or []
    assert sum(1 for r in h_hi if r.get("relocated")) > 0, "the loose arm never relocated, so it tests nothing"
    assert any("interp_error_p90" in r for r in h_hi), "the indicator was never measured"

    n_hi = sum(1 for r in h_hi if r.get("escalated"))
    n_lo = sum(1 for r in h_lo if r.get("escalated"))
    assert n_hi == 0, f"a tolerance of 50 (the field range is ~1) must never escalate, got {n_hi}"
    assert n_lo > 0, "a tolerance of 0.02 must escalate at least once"

    n0 = len(np.asarray(hi.meshes[0][0]))
    assert len(np.asarray(hi.meshes[-1][0])) == n0, "relocation alone must hold the node count"
    assert len(np.asarray(lo.meshes[-1][0])) > n0, "an escalation must actually add vertices"


def test_escalation_is_refused_without_a_sane_tolerance():
    with pytest.raises(ValueError, match="escalate"):
        jno.solve.relocate(escalate=0.0)
    with pytest.raises(ValueError, match="escalate_growth"):
        jno.solve.relocate(escalate=0.1, escalate_growth=1.0)
