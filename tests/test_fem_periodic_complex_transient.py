"""Periodic ties composed with a complex *transient* FEM (Phase 6 of the compose work).

A complex transient (e.g. Schrodinger/complex-diffusion) integrates the real-equivalent 2N time block
``[[M_r,-M_i],[M_i,M_r]]`` etc. A periodic tie reduces *both* real/imag blocks by the same ``P`` (the
combinator recurses into the ``(block_r, block_i)`` legs), the integration runs in the reduced main-
DOF space, and each saved slice is prolonged back -- real/imag separately. (Previously this raised
NotImplementedError.) Pinned with a manufactured decaying periodic mode and a zero-IC extreme.

Run with x64 (the trajectory is complex128).
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


def _build(amp_ic, mesh_size=0.07):
    """Complex diffusion ``ψ_t = c Δψ`` (c = 0.5+1j), periodic in x + Dirichlet y. Manufactured periodic
    mode ``ψ0 = amp_ic · cos(2πx) sin(πy)`` decays as ``ψ(t) = exp(-c·5π²·t) ψ0`` (``-Δψ0 = 5π² ψ0``)."""
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
    c = 0.5 + 1j
    psi0 = amp_ic * jno.np.cos(2 * PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem(
        [
            ui.t * vi + c * (ui.x * vi.x + ui.y * vi.y),
            u(xb, yb) - 0.0,
            u(xt, yt) - 0.0,
            u(xl, yl) - u(xr, yr),  # periodic in x
            u(ci[0], ci[1]) - psi0,
        ]
    )
    return fem, c


def test_periodic_complex_transient_recovers_decaying_mode():
    """The periodic complex-diffusion strip recovers the analytic decaying mode, with the tie reducing
    the 2N time block."""
    fem, c = _build(amp_ic=1.0)
    assert fem.is_complex and fem.is_transient
    assert fem._periodic is not None, "the u(left)-u(right) tie must reduce the complex transient block"
    assert fem._periodic["n_red"] < fem._periodic["n_full"], "the tie must eliminate the secondary-face DOFs"

    traj = np.asarray(fem.solve())
    assert np.iscomplexobj(traj)
    pts = np.asarray(fem.points)
    mode = np.cos(2 * PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    t1 = float(fem.t1)
    analytic = np.exp(-c * 5 * PI**2 * t1) * mode
    rel = float(np.linalg.norm(traj[-1] - analytic) / np.linalg.norm(analytic))
    assert rel < 5e-2, f"periodic complex transient recovery rel-L2 {rel:.3e}"
    assert float(np.abs(traj[-1].imag).max()) > 1e-2, "must be genuinely complex"


def test_periodic_complex_transient_zero_ic_is_zero():
    """Extreme: a zero IC stays identically zero (homogeneous, no source) -- the reduction/prolongation
    must not inject a spurious field on either the real or the imaginary block."""
    fem, _ = _build(amp_ic=0.0)
    traj = np.asarray(fem.solve())
    assert float(np.abs(traj).max()) < 1e-10, (
        f"zero-IC periodic complex transient must stay 0, got {np.abs(traj).max():.2e}"
    )
