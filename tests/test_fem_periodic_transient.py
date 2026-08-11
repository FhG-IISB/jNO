"""Real (non-complex) periodic transient: the tie reduces the semidiscrete block, so ``fem.solve()`` must
return the trajectory on the FULL nodal layout -- prolonged ``u = P·u_red`` -- not the reduced main DOFs.

Every steady operator (``FemLinearSystem.solve``) and the complex-transient path already prolong; the real
transient path did **not**, so a periodic transient handed back a reduced trajectory that a caller then
mis-sliced with full-mesh offsets. That is exactly the bug that scrambled the 3-D ``jno.litho.CAResist``
reaction-diffusion PEB (its inhibitor block was read with un-reduced offsets). Pinned here with a
manufactured decaying periodic heat mode whose analytic value lives on the full mesh -- the trajectory only
matches (and is only the right length) once prolonged.
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402
from jno.trace_evaluator import TraceEvaluator  # noqa: E402

PI = np.pi


def _traj(fem):
    """Evaluate a (non-parametric) transient ``fem.solve()`` trace node to its ``(n_step, n_dof)`` array."""
    node = fem.solve()
    return np.asarray(TraceEvaluator({}).evaluate(node.expr if hasattr(node, "expr") else node, context={}))


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _build(amp_ic, mesh_size=0.07):
    """Heat ``u_t = c Δu`` (c = 0.5), periodic in x + Dirichlet ``u = 0`` on y. Manufactured periodic mode
    ``u0 = amp_ic · cos(2πx) sin(πy)`` decays as ``u(t) = exp(-c·5π²·t) u0``  (``-Δu0 = 5π² u0``)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size, time=(0.0, 0.02, 41))
    d.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    d.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    d.tag("bottom", lambda x, y: y < 1e-6)
    d.tag("top", lambda x, y: y > 1 - 1e-6)
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("bottom", split=True)
    xt, yt, _ = d.variable("top", split=True)
    xl, yl, _ = d.variable("left", split=True)
    xr, yr, _ = d.variable("right", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    c = 0.5
    u0 = amp_ic * jno.np.cos(2 * PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem(
        [
            ui.t * vi + c * (ui.x * vi.x + ui.y * vi.y),
            u(xb, yb) - 0.0,
            u(xt, yt) - 0.0,
            u(xl, yl) - u(xr, yr),  # periodic in x
            u(ci[0], ci[1]) - u0,
        ]
    )
    return fem, c


def test_periodic_transient_returns_full_prolonged_trajectory():
    """The tie reduces the block (``n_red < n_full``), and ``fem.solve()`` must return the trajectory on the
    FULL mesh -- one column per ``fem.points`` node -- recovering the analytic decaying mode. Without the
    prolongation the trajectory has the reduced length and cannot match the full-mesh analytic field."""
    fem, c = _build(amp_ic=1.0)
    assert fem.is_transient and fem._periodic is not None
    assert fem._periodic["n_red"] < fem._periodic["n_full"], "the u(left)-u(right) tie must eliminate secondary DOFs"

    traj = _traj(fem)
    pts = np.asarray(fem.points)
    # the returned trajectory lives on the FULL nodal layout (prolonged), not the reduced main DOFs
    assert traj.shape[1] == pts.shape[0] == fem._periodic["n_full"], (
        f"trajectory width {traj.shape[1]} must be the full node count {fem._periodic['n_full']} (prolonged), "
        f"not the reduced {fem._periodic['n_red']}"
    )
    mode = np.cos(2 * PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    analytic = np.exp(-c * 5 * PI**2 * float(fem.t1)) * mode
    rel = float(np.linalg.norm(traj[-1] - analytic) / np.linalg.norm(analytic))
    assert rel < 5e-2, f"periodic transient recovery rel-L2 {rel:.3e}"


def test_periodic_transient_ties_left_to_right_after_prolong():
    """The prolonged field satisfies the periodic tie itself: at every saved step the left-face values equal
    the right-face values (the eliminated secondary DOFs were reconstructed from their mains, ``u = P·u_red``).
    A reduced (un-prolonged) trajectory has no left/right columns at all to compare."""
    fem, _ = _build(amp_ic=1.0)
    traj = _traj(fem)
    pts = np.asarray(fem.points)
    left = np.where(pts[:, 0] < 1e-6)[0]
    right = np.where(pts[:, 0] > 1 - 1e-6)[0]
    lo = left[np.argsort(pts[left, 1])]  # order both faces by y so the main/secondary pairs line up
    ro = right[np.argsort(pts[right, 1])]
    assert lo.size > 1 and lo.size == ro.size
    np.testing.assert_allclose(traj[:, lo], traj[:, ro], atol=1e-8)


def test_periodic_transient_zero_ic_stays_zero():
    """Extreme: a zero initial condition stays identically zero (homogeneous heat, no source) -- the
    reduction/prolongation must not inject a spurious field on the full mesh."""
    fem, _ = _build(amp_ic=0.0, mesh_size=0.1)
    traj = _traj(fem)
    assert float(np.abs(traj).max()) < 1e-10, f"zero-IC periodic transient must stay 0, got {np.abs(traj).max():.2e}"
