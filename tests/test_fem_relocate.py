"""r-adaptivity via the adapt slot — ``FEM.solve(adapt=AdaptSpec(relocate=True))``.

Relocates the mesh vertices tagged with :meth:`Variable.trainable` down the FE-energy gradient (through the
differentiable solve) with a backtracking mesh-validity line search — the built-in companion of h-refinement
(``run_adaptive_relocate``). Checks that it reduces the objective at **fixed DOF** without tangling across
**scalar, vector, nonlinear, transient, periodic, and complex** problems, and demands at least one
``.trainable()`` coordinate.
"""

import jax
import numpy as np
import pytest

import jno
import jno.jnp_ops as J
from jno.utils.solver.fem_adapt import AdaptSpec


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _min_detj(pts, cells):
    v = pts[cells]
    a, b = v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]
    return float(np.min(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]))


def _peak_scalar(size=0.14, movable=True):
    """Poisson with a sharp off-center peak source; interior nodes (a central box) tagged trainable."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    if movable:
        xm, ym, _ = d.variable("mov", where=lambda x, y: (x > 0.2) & (x < 0.8) & (y > 0.2) & (y < 0.8), split=True)
        xm.trainable(name="ix")
        ym.trainable(name="iy")
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = J.exp(-40.0 * ((xi - 0.62) ** 2 + (yi - 0.35) ** 2))
    return d, jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)


def test_relocate_reduces_energy_and_stays_valid():
    d, fem = _peak_scalar()
    pts0 = np.asarray(d.mesh.points)[:, :2].copy()
    n0 = len(pts0)
    sol = np.asarray(fem.solve(adapt=AdaptSpec(relocate=True, max_iters=40, lr=3e-3, quality_floor=0.1))).reshape(-1)
    hist = fem.adapt_history
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    pts_r = np.asarray(fem.domain.mesh.points)[:, :2]

    assert len(hist) >= 5, "relocation should take several steps"
    assert hist[-1]["energy"] < hist[0]["energy"], "relocation must reduce the FE energy"
    assert len(pts_r) == n0 and sol.shape[0] == n0, "r-adaptivity adds no DOFs (fixed connectivity)"
    assert _min_detj(pts_r, cells) > 0.0, "the relocated mesh must stay valid (no inverted elements)"
    assert np.linalg.norm(pts_r - pts0) > 1e-3, "the interior vertices should actually move toward the feature"


def test_relocate_requires_trainable_coordinates():
    _, fem = _peak_scalar(movable=False)
    with pytest.raises(ValueError, match="no trainable mesh coordinates"):
        fem.solve(adapt=AdaptSpec(relocate=True))


def test_relocate_vector_field():
    """Generality: a vector problem relocates too (the energy objective sums over components)."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.16).domain()
    u, phi = d.fem_symbols(value_shape=(2,))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xm, ym, _ = d.variable("mov", where=lambda x, y: (x > 0.2) & (x < 0.8) & (y > 0.2) & (y < 0.8), split=True)
    xm.trainable(name="ix")
    ym.trainable(name="iy")
    vi = phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
    weak = jno.np.inner(jno.np.grad(u, [xi, yi]), jno.np.grad(phi, [xi, yi]), n_contract=2) - (
        f * vi.component(0) + 0.5 * f * vi.component(1)
    )
    fem = jno.fem([weak, u(xb, yb) - 0.0])
    n0 = len(d.mesh.points)
    sol = np.asarray(fem.solve(adapt=AdaptSpec(relocate=True, max_iters=25, lr=2e-3))).reshape(-1)
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    assert fem.adapt_history[-1]["energy"] <= fem.adapt_history[0]["energy"], "vector relocation should not raise energy"
    assert _min_detj(np.asarray(fem.domain.mesh.points)[:, :2], cells) > 0.0, "vector relocation must stay valid"
    assert sol.shape[0] == 2 * n0, "vector solution has 2 DOFs per node, unchanged by relocation"


def _mov(d):
    xm, ym, _ = d.variable("mov", where=lambda x, y: (x > 0.2) & (x < 0.8) & (y > 0.2) & (y < 0.8), split=True)
    xm.trainable(name="ix")
    ym.trainable(name="iy")


def test_relocate_nonlinear():
    """A steady *nonlinear* problem relocates (the objective's solve is a differentiable Newton solve)."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.18).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    _mov(d)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 10.0 * J.exp(-40.0 * ((xi - 0.6) ** 2 + (yi - 0.35) ** 2))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + ui * ui * ui * vi - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    assert fem._mode == "nonlinear"
    fem.solve(adapt=AdaptSpec(relocate=True, max_iters=20, lr=2e-3))
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    assert fem.adapt_history[-1]["energy"] <= fem.adapt_history[0]["energy"]
    assert _min_detj(np.asarray(fem.domain.mesh.points)[:, :2], cells) > 0.0


def test_relocate_transient():
    """A *transient* problem relocates for the whole trajectory (time-averaged energy; the coord gradient
    flows through the marched block)."""
    from shapely.geometry import box

    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.18, time=(0.0, 0.3, 11))
    u, v = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    _mov(d)
    ui, vi = u.bind(x=xi, y=yi, t=ti), v.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0, u(ci[0], ci[1]) - 1.0])
    assert fem.is_transient
    fem.solve(adapt=AdaptSpec(relocate=True, max_iters=15, lr=2e-3))
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    h = fem.adapt_history
    assert h[-1]["energy"] < h[0]["energy"], "transient relocation should reduce the time-averaged energy"
    assert _min_detj(np.asarray(fem.domain.mesh.points)[:, :2], cells) > 0.0


def test_relocate_periodic():
    """A *periodic* problem relocates: interior relocation never touches the boundary ties."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.18).domain()
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("left", where=lambda x, y: x < 1e-6, split=True)
    xr, yr, _ = d.variable("right", where=lambda x, y: x > 1 - 1e-6, split=True)
    bt = d.variable("bt", where=lambda x, y: (y < 1e-6) | (y > 1 - 1e-6), split=True)
    _mov(d)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = J.exp(-40.0 * ((xi - 0.5) ** 2 + (yi - 0.35) ** 2))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xl, yl) - u(xr, yr), u(bt[0], bt[1]) - 0.0])
    fem.solve(adapt=AdaptSpec(relocate=True, max_iters=20, lr=2e-3))
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    assert fem.adapt_history[-1]["energy"] <= fem.adapt_history[0]["energy"]
    assert _min_detj(np.asarray(fem.domain.mesh.points)[:, :2], cells) > 0.0


def test_relocate_complex():
    """A *complex* problem relocates: complex is two real blocks (real + imag), and the energy sums both."""
    from shapely.geometry import box

    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.16)
    u, w = d.fem_symbols(complex=True)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    _mov(d)
    ub, wb = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)
    c = 1.0 + 0.5j
    f = jno.complex(
        10.0 * J.exp(-40.0 * ((xi - 0.6) ** 2 + (yi - 0.35) ** 2)),
        8.0 * J.exp(-40.0 * ((xi - 0.35) ** 2 + (yi - 0.6) ** 2)),
    )
    weak = (ub.x * wb.x + ub.y * wb.y) - c * (ub * wb) - f * wb
    fem = jno.fem([weak.real, u.real(xb, yb) - 0.0, u.imag(xb, yb) - 0.0])
    assert fem._mode == "linear" and len(fem.offsets) == 3  # a real 2N block system (real + imag)
    sol = np.asarray(fem.solve(adapt=AdaptSpec(relocate=True, max_iters=20, lr=2e-3))).reshape(-1)
    cells = np.asarray(fem.domain.mesh.cells_dict["triangle"])
    n0 = len(fem.domain.mesh.points)
    assert fem.adapt_history[-1]["energy"] <= fem.adapt_history[0]["energy"]
    assert _min_detj(np.asarray(fem.domain.mesh.points)[:, :2], cells) > 0.0
    assert sol.shape[0] == 2 * n0, "complex solution = real + imaginary blocks"
