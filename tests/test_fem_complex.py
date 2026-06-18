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

    u_num = np.asarray(fem.solve())
    assert np.iscomplexobj(u_num)
    pts = np.asarray(fem.points)
    u_star = amp * np.cos(PI * pts[:, 0]) * np.cos(PI * pts[:, 1])
    rel = float(np.linalg.norm(u_num - u_star) / np.linalg.norm(u_star))
    assert rel < 1e-2, f"complex Helmholtz recovery rel-L2 {rel:.3e}"
    assert float(np.abs(u_num.imag).max()) > 0.1  # genuinely complex, not a real solve in disguise
