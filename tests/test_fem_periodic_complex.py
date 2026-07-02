"""Periodic ties composed with a complex-valued FEM (Phase 2 of the compose work).

A complex weak form is assembled as two real systems and solved via the real-equivalent block
``[[A_r,-A_i],[A_i,A_r]]``. A periodic tie reduces the system with a prolongation ``P``. The two
compose: because Re and Im share one FE space (hence one ``P``), reducing *both* legs with the same
``P`` preserves the real-equivalent block exactly, ``blkdiag(P,P)^T[[A_r,-A_i],[A_i,A_r]]blkdiag(P,P)``.
These tests pin the composition with manufactured complex solutions that are genuinely periodic.

Run with x64 (the solution is complex128).
"""

import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

PI = np.pi


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_periodic_complex_helmholtz_recovers_manufactured():
    """Complex Helmholtz ``c(-Δu) + d·u = f`` periodic in x (``u(left)-u(right)``) and Dirichlet in y.
    Manufactured ``u* = (1+0.5i) cos(2πx) sin(πy)`` -- periodic in x (matching value AND flux at the
    tie), zero on y=0,1. With ``-Δu* = 5π² u*``, ``f = (5π² c + d) u*``. The complex solve runs through
    the periodic reduction; recovery of the (genuinely complex) ``u*`` confirms the composition."""
    dom = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    # periodic faces are the OPEN edges (corners belong to the Dirichlet top/bottom, not the tie)
    dom.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("bottom", lambda x, y: y < 1e-6)
    dom.tag("top", lambda x, y: y > 1 - 1e-6)

    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

    sigma = 0.5 + 0.0 * xi  # traced complex coefficient (stresses complex division through the trace)
    c = 1.0 / (1.0 + 1j * sigma)
    d_coef = 1.0 + 0.2j  # nonzero reaction -> the periodic-in-x / Dirichlet-in-y problem is well-posed
    amp = 1.0 + 0.5j
    g = jno.np.cos(2 * PI * xi) * jno.np.sin(PI * yi)
    f = (5 * PI**2 * c + d_coef) * amp * g

    fem = jno.fem(
        [
            c * (ui.x * vi.x + ui.y * vi.y) + d_coef * (u * vi) - f * vi,
            u(xb, yb) - 0.0,
            u(xt, yt) - 0.0,
            u(xl, yl) - u(xr, yr),  # periodic in x
        ]
    )
    assert fem.is_complex
    assert fem.problem is None  # native real-equivalent assembly
    assert fem._periodic is not None, "the u(left)-u(right) tie must reduce the complex system"
    assert fem._periodic["n_red"] < fem._periodic["n_full"], "the tie must eliminate the slave-face DOFs"

    u_num = np.asarray(fem.solve())
    assert np.iscomplexobj(u_num)
    pts = np.asarray(fem.points)
    u_star = amp * np.cos(2 * PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    rel = float(np.linalg.norm(u_num - u_star) / np.linalg.norm(u_star))
    assert rel < 2e-2, f"periodic complex Helmholtz recovery rel-L2 {rel:.3e}"
    assert float(np.abs(u_num.imag).max()) > 0.1, "must be genuinely complex, not a real solve in disguise"


def test_periodic_complex_homogeneous_is_zero():
    """Extreme: zero source + homogeneous Dirichlet + periodic. The reaction term makes the operator
    nonsingular, so the only solution is ``u ≡ 0`` -- the reduction must not inject a spurious field."""
    dom = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    dom.tag("left", lambda x, y: (x < 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("right", lambda x, y: (x > 1 - 1e-6) & (y > 1e-6) & (y < 1 - 1e-6))
    dom.tag("bottom", lambda x, y: y < 1e-6)
    dom.tag("top", lambda x, y: y > 1 - 1e-6)

    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    c = 1.0 / (1.0 + 1j * (0.5 + 0.0 * xi))

    fem = jno.fem(
        [
            c * (ui.x * vi.x + ui.y * vi.y) + (1.0 + 0.2j) * (u * vi),  # no source
            u(xb, yb) - 0.0,
            u(xt, yt) - 0.0,
            u(xl, yl) - u(xr, yr),
        ]
    )
    assert fem.is_complex and fem._periodic is not None
    u_num = np.asarray(fem.solve())
    assert float(np.abs(u_num).max()) < 1e-9, f"homogeneous periodic complex must be ~0, got {np.abs(u_num).max():.2e}"


def test_doubly_periodic_complex_reaction_diffusion():
    """Extreme: two tied face-pairs (a doubly-periodic complex cell). ``c(-Δu) + d·u = f`` periodic in
    **both** x and y -- the four corners are each a slave in two directions and must collapse onto one
    kept master (transitive corner resolution) *through* the complex real-equivalent block. Manufactured
    ``u* = (1+0.5i) cos(2πx) cos(2πy)`` with ``-Δu* = 8π² u*``."""
    dom = jno.domain(box(0.0, 0.0, 1.0, 1.0)).build_mesh(0.06)
    for nm, pred in {
        "left": lambda x, y: x < 1e-6,
        "right": lambda x, y: x > 1 - 1e-6,
        "bottom": lambda x, y: y < 1e-6,
        "top": lambda x, y: y > 1 - 1e-6,
    }.items():
        dom.tag(nm, pred)

    u, phi = dom.fem_symbols()
    xi, yi, _ = dom.variable("interior", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

    c = 1.0 / (1.0 + 1j * (0.5 + 0.0 * xi))
    d_coef = 1.0 + 0.2j
    amp = 1.0 + 0.5j
    g = jno.np.cos(2 * PI * xi) * jno.np.cos(2 * PI * yi)
    f = (8 * PI**2 * c + d_coef) * amp * g

    fem = jno.fem(
        [
            c * (ui.x * vi.x + ui.y * vi.y) + d_coef * (u * vi) - f * vi,
            u(xl, yl) - u(xr, yr),  # periodic in x
            u(xb, yb) - u(xt, yt),  # periodic in y
        ]
    )
    assert fem.is_complex and fem._periodic is not None and fem._periodic["n_red"] < fem._periodic["n_full"]
    u_num = np.asarray(fem.solve())
    pts = np.asarray(fem.points)
    u_star = amp * np.cos(2 * PI * pts[:, 0]) * np.cos(2 * PI * pts[:, 1])
    rel = float(np.linalg.norm(u_num - u_star) / np.linalg.norm(u_star))
    assert rel < 5e-2, f"doubly-periodic complex reaction-diffusion rel-L2 {rel:.3e}"
    assert float(np.abs(u_num.imag).max()) > 0.1
