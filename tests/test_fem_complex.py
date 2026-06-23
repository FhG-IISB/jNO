"""Complex-valued FEM through the real-equivalent block (feax assembled real-only).

``jno.fem`` detects a complex weak form, splits each term into real Re/Im sub-forms
(``Re(c·T)=Re(c)·T`` since the FE trial/test ``T`` is real), assembles both through the ordinary
**real** feax path, solves the real block ``[[A_r,-A_i],[A_i,A_r]]``, and returns ``u_r + i·u_i``.
No feax change, no reliance on feax's native-complex behavior.

Run with x64 (the solution is complex128): ``JAX_ENABLE_X64=1``.
"""

import pytest

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import numpy as np  # noqa: E402
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


def test_complex_helmholtz_real_equivalent_recovers_manufactured():
    """Manufactured complex Helmholtz, all-Neumann (no Dirichlet bookkeeping):
        c(-lap u) + d u = f,  c = 1/(1 + i sigma) (complex division *through the trace*),
        u* = (1 + 0.5i) cos(pi x) cos(pi y)  (zero normal derivative on the box),
        f = (2 pi^2 c + d) u*.
    The real-equivalent block recovers u* (the operator AND the source are complex; both are
    assembled as real Re/Im sub-forms)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)

    sigma = 0.5 + 0.0 * xi  # traced -> c is a *traced* complex expression (stresses complex division)
    c = 1.0 / (1.0 + 1j * sigma)
    d_coef = -(1.0 + 0.2j)
    amp = 1.0 + 0.5j
    g = jno.np.cos(PI * xi) * jno.np.cos(PI * yi)
    f = (2 * PI**2 * c + d_coef) * amp * g

    fem = jno.fem([c * (ui.x * vi.x + ui.y * vi.y) + d_coef * (u * vi) - f * vi])
    assert fem.is_complex
    assert fem.problem is None  # the Re/Im real systems are assembled natively (no feax problem)

    u_num = np.asarray(fem.solve())
    assert np.iscomplexobj(u_num)
    pts = np.asarray(fem.points)
    u_star = amp * np.cos(PI * pts[:, 0]) * np.cos(PI * pts[:, 1])
    rel = float(np.linalg.norm(u_num - u_star) / np.linalg.norm(u_star))
    assert rel < 1e-2, f"complex Helmholtz recovery rel-L2 {rel:.3e}"
    assert float(np.abs(u_num.imag).max()) > 0.1  # genuinely complex, not a real solve in disguise


def test_pml_helmholtz_absorbs_reflection_free():
    """2D Helmholtz with a perfectly-matched layer (PML) -- the headline use case. The complex
    coordinate stretch ``s = 1 + i sigma/k`` (sigma ramps in a frame, 0 in the physical core)
    absorbs outgoing waves; the outer wall is u=0. The imaginary unit is Python's native ``1j``.

    PML-quality gate = sigma-insensitivity: a *converged* PML's physical-core solution does not
    depend on the absorber strength (a poor/absent PML would reflect and change with sigma)."""
    L, w, k = 1.0, 0.3, 12.0
    relu = lambda z: jno.np.maximum(z, 0.0)  # noqa: E731

    def solve_pml(sigma0):
        dom = jno.domain(box(0.0, 0.0, L, L), mesh_size=0.045)
        u, phi = dom.fem_symbols()
        xi, yi, _ = dom.variable("interior", split=True)
        xb, yb, _ = dom.variable("boundary", split=True)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        sx = sigma0 * (relu(w - xi) ** 2 + relu(xi - (L - w)) ** 2) / w**2  # per-axis PML depth
        sy = sigma0 * (relu(w - yi) ** 2 + relu(yi - (L - w)) ** 2) / w**2
        Sx, Sy = 1.0 + 1j * sx / k, 1.0 + 1j * sy / k  # complex coordinate stretch
        src = jno.np.exp(-(((xi - 0.5) ** 2 + (yi - 0.5) ** 2) / (2 * 0.03**2)))  # ~point source
        weak = (Sy / Sx) * (ui.x * vi.x) + (Sx / Sy) * (ui.y * vi.y) - k**2 * Sx * Sy * (u * vi) - src * vi
        fem = jno.fem([weak, u(xb, yb) - 0.0], quad_degree=3)
        return fem, np.asarray(fem.solve()), np.asarray(fem.points)

    fem, u1, pts = solve_pml(40.0)
    _, u2, _ = solve_pml(60.0)  # 1.5x absorber strength, fresh mesh
    assert fem.is_complex and np.iscomplexobj(u1) and not bool(np.isnan(u1).any())
    assert fem.problem is None  # native real-equivalent assembly (Dirichlet wall included), no feax

    core = (pts[:, 0] > w) & (pts[:, 0] < L - w) & (pts[:, 1] > w) & (pts[:, 1] < L - w)
    sigma_insens = float(np.linalg.norm(u1[core] - u2[core]) / np.linalg.norm(u1[core]))
    assert sigma_insens < 1e-2, f"PML not reflection-free: sigma-insensitivity {sigma_insens:.3e}"
    assert float(np.abs(u1[core].imag).max()) > 1e-3  # a propagating (complex) wave, not a static field


def test_complex_transient_recovers_mode_and_conserves_schrodinger_norm():
    """Complex *transient* FEM via the real-equivalent block (the M, A, and IC are each split into
    real Re/Im parts; backward Euler runs on the ``2N`` real block ``[[M_r,-M_i],[M_i,M_r]]`` etc.).

        psi_t = c lap psi   on the unit square, psi = 0 walls,
        IC psi0 = sin(pi x) sin(pi y)  (real)  ->  psi(t) = exp(-c 2 pi^2 t) psi0.

    Two regimes from the *same* machinery:
      * c = 0.5 + 1j : a complex diffusion (decay + oscillation), recovered vs the analytic mode.
      * c = 1j       : free-particle Schrodinger (i psi_t = -lap psi) -- unitary, so |psi| is
                       conserved; backward Euler is only mildly dissipative."""

    def solve(c):
        d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.07, time=(0.0, 0.05, 51))
        u, phi = d.fem_symbols()
        xi, yi, ti = d.variable("interior", split=True)
        xb, yb, _ = d.variable("boundary", split=True)
        ci = d.variable("initial", split=True)
        ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
        psi0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])  # real IC; the dynamics make psi complex
        fem = jno.fem([ui.t * vi + c * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - psi0])
        return fem, np.asarray(fem.solve()), np.asarray(fem.points)

    # complex diffusion: decay + oscillation, checked against the analytic mode
    fem, traj, pts = solve(0.5 + 1j)
    assert fem.is_complex and fem.is_transient and np.iscomplexobj(traj)
    t1 = float(fem.t1)
    mode = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    analytic = np.exp(-(0.5 + 1j) * 2 * PI**2 * t1) * mode
    rel = float(np.linalg.norm(traj[-1] - analytic) / np.linalg.norm(analytic))
    assert rel < 3e-2, f"complex transient recovery rel-L2 {rel:.3e}"
    assert float(np.abs(traj[-1].imag).max()) > 1e-2  # genuinely complex trajectory, not a real solve

    # Schrodinger free particle: unitary -> |psi| conserved (BE only mildly dissipative)
    fem_s, traj_s, pts_s = solve(1j)
    mode_s = np.sin(PI * pts_s[:, 0]) * np.sin(PI * pts_s[:, 1])
    rel_s = float(np.linalg.norm(traj_s[-1] - np.exp(-1j * 2 * PI**2 * t1) * mode_s) / np.linalg.norm(mode_s))
    assert rel_s < 3e-2, f"Schrodinger recovery rel-L2 {rel_s:.3e}"
    ratio = float(np.linalg.norm(traj_s[-1]) / np.linalg.norm(traj_s[0]))
    assert 0.97 < ratio < 1.01, f"Schrodinger norm not conserved: |psi(t1)|/|psi(0)| {ratio:.4f}"
