"""Complex FEM the **front-end** way: ``domain.fem_symbols(..., complex=True)``.

A complex field is one symbol — a complex trial/test — written with ordinary complex algebra
(``*`` is the complex product, ``1j``, ``.real``/``.imag``, ``.dot``). Under the hood it is carried
as two coupled real fields ``(u_r, u_i)``; ``jno.fem`` lowers ``weak.real`` onto the coupled
multifield real system (the distribution fallback splits the grouped real part per test field). These
tests pin the manufactured recovery for a scalar complex Helmholtz (Re != Im, genuinely complex
coefficient — the discriminating case) and the 2-D vector Maxwell curl-curl problem.

Run with x64 (the coupled real system is float64): ``JAX_ENABLE_X64=1``.
"""

import pytest

pytest.importorskip("feax", reason="feax required for FEM assembly")
pytest.importorskip("shapely", reason="shapely required for the box domain")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    """The coupled real system is float64, so these tests need x64. Set it *per-test* with
    save/restore so the global flag never leaks into modules co-run after this one (a
    module-level ``update`` runs at collection time and pollutes the whole session)."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _solve_dense(fem):
    """Solve the assembled steady linear block densely (small test meshes)."""
    A = np.asarray(fem.A.todense() if hasattr(fem.A, "todense") else fem.A)
    b = np.asarray(fem.b).reshape(-1)
    return np.linalg.solve(A, b), int(np.asarray(fem.problem.offset)[1]), np.asarray(fem.problem.mesh[0].points)


def test_complex_true_returns_complex_pair():
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.4)
    u, w = d.fem_symbols(complex=True)
    assert isinstance(u, jno.ComplexPair) and isinstance(w, jno.ComplexPair)
    assert isinstance(u.real, jno.TrialFunction) and isinstance(u.imag, jno.TrialFunction)
    assert isinstance(w.real, jno.TestFunction) and isinstance(w.imag, jno.TestFunction)


def test_complex_scalar_helmholtz_via_complex_true():
    """-Δu - c·u = f with c = 1 + 0.5i and a manufactured u where Re(u) != Im(u)."""
    pi, sin = np.pi, jno.np.sin
    cr, ci = 1.0, 0.5
    c = cr + 1j * ci
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.07)
    u, w = d.fem_symbols(complex=True)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ub, wb = u.bind(x=xi, y=yi), w.bind(x=xi, y=yi)

    URi = sin(pi * xi) * sin(pi * yi)  # -Δ = 2π²
    UIi = sin(2 * pi * xi) * sin(pi * yi)  # -Δ = 5π²  (Re != Im)
    fr = 2 * pi**2 * URi - (cr * URi - ci * UIi)
    fi = 5 * pi**2 * UIi - (cr * UIi + ci * URi)
    f = jno.complex(fr, fi)

    weak = (ub.x * wb.x + ub.y * wb.y) - c * (ub * wb) - f * wb  # complex; `*` is the complex product
    fem = jno.fem([weak.real, u.real(xb, yb) - 0.0, u.imag(xb, yb) - 0.0])
    assert fem._mode == "linear"  # a complex *linear* form must not be misread as nonlinear

    sol, n, pts = _solve_dense(fem)
    URn = np.sin(pi * pts[:, 0]) * np.sin(pi * pts[:, 1])
    UIn = np.sin(2 * pi * pts[:, 0]) * np.sin(pi * pts[:, 1])
    assert np.linalg.norm(sol[:n] - URn) / np.linalg.norm(URn) < 5e-3
    assert np.linalg.norm(sol[n:] - UIn) / np.linalg.norm(UIn) < 5e-3


def test_complex_vector_maxwell_via_complex_true():
    """Time-harmonic curl-curl ``curl(curl E) - k²E = J`` with complex ``k²`` and ``Re(E) != Im(E)``."""
    pi, sin, cos = np.pi, jno.np.sin, jno.np.cos
    KR, KI = 30.0, 4.0
    k2 = KR + 1j * KI

    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.1)
    E, v = d.fem_symbols(value_shape=(2,), names=("E", "v"), order=2, complex=True)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    Eb, vb = E.bind(x=xi, y=yi), v.bind(x=xi, y=yi)

    curl = lambda F: F.x[1] - F.y[0]  # noqa: E731
    div = lambda F: F.x[0] + F.y[1]  # noqa: E731
    s = KR  # grad-div penalty (consistent: exact E is divergence-free)

    exr, eyr = pi * sin(pi * xi) * cos(pi * yi), -pi * cos(pi * xi) * sin(pi * yi)
    exi, eyi = 2 * pi * sin(2 * pi * xi) * cos(2 * pi * yi), -2 * pi * cos(2 * pi * xi) * sin(2 * pi * yi)
    jxr, jyr = (2 * pi**2 - KR) * exr + KI * exi, (2 * pi**2 - KR) * eyr + KI * eyi
    jxi, jyi = (8 * pi**2 - KR) * exi - KI * exr, (8 * pi**2 - KR) * eyi - KI * eyr
    Jx, Jy = jno.complex(jxr, jxi), jno.complex(jyr, jyi)

    weak = curl(Eb) * curl(vb) + s * div(Eb) * div(vb) - k2 * Eb.dot(vb) - (Jx * vb[0] + Jy * vb[1])
    brx, bry = pi * sin(pi * xb) * cos(pi * yb), -pi * cos(pi * xb) * sin(pi * yb)
    bix, biy = 2 * pi * sin(2 * pi * xb) * cos(2 * pi * yb), -2 * pi * cos(2 * pi * xb) * sin(2 * pi * yb)
    fem = jno.fem(
        [
            weak.real,
            E.real(xb, yb)[0] - brx,
            E.real(xb, yb)[1] - bry,
            E.imag(xb, yb)[0] - bix,
            E.imag(xb, yb)[1] - biy,
        ],
        quad_degree=6,
    )
    assert fem._mode == "linear"

    sol, n, pts = _solve_dense(fem)
    E_re, E_im = sol[:n].reshape(-1, 2), sol[n:].reshape(-1, 2)
    px, py = pts[:, 0], pts[:, 1]
    ex_re = np.stack([pi * np.sin(pi * px) * np.cos(pi * py), -pi * np.cos(pi * px) * np.sin(pi * py)], 1)
    ex_im = np.stack(
        [2 * pi * np.sin(2 * pi * px) * np.cos(2 * pi * py), -2 * pi * np.cos(2 * pi * px) * np.sin(2 * pi * py)], 1
    )
    rel = np.linalg.norm(np.concatenate([E_re - ex_re, E_im - ex_im])) / np.linalg.norm(np.concatenate([ex_re, ex_im]))
    assert rel < 5e-3, f"Maxwell not recovered: {rel:.2e}"  # coarse 0.1 mesh; the tutorial (0.06) hits 5e-4
