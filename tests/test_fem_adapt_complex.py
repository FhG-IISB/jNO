"""Adaptive FEM (``jno.solve(adapt=)``) on **complex** fields.

The ZZ recovery estimator drives ``solve -> estimate -> mark -> refine``. Historically it was
scalar-real-P1 only: a complex field passed the vertex-count guard but the real-only gradient math
silently dropped the imaginary part, so refinement was driven by ``Re(u)`` alone. These tests pin
the complex path: the indicator uses the modulus of the (complex) gradient gap -- so BOTH real and
imaginary variation drive refinement -- while staying byte-identical on real fields (``|x|^2==x^2``).

Complex assembly runs in float64, so x64 is forced here.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("mmgpy", reason="mmgpy required for adaptive remeshing")
pytest.importorskip("shapely", reason="shapely required for PolygonDomain")

import jax  # noqa: E402
from shapely.geometry import box  # noqa: E402

from jno.utils.solver.fem_adapt import zz_error_indicators  # noqa: E402
from jno.utils.solver.linear import sparse_lu_solve  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_zz_indicator_uses_the_imaginary_part():
    """Discriminating unit test: a field whose ONLY sharp feature is in ``Im(u)`` (``Re(u)`` is
    linear, hence ZZ-exact) must still be flagged by the estimator -- proving the imaginary part is
    not discarded. A purely-smooth real field gives a ~0 estimate (so the flag came from ``Im``)."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.05)
    pts = np.asarray(d.mesh.points)
    x, y = pts[:, 0], pts[:, 1]
    re = 0.3 * x + 0.2 * y  # linear -> zero recovery error
    im = np.exp(-((x - 0.7) ** 2 + (y - 0.3) ** 2) / (2 * 0.05**2))  # sharp bump in the imaginary part

    eta, est = zz_error_indicators(d, re + 1j * im)
    eta_smooth, est_smooth = zz_error_indicators(d, re.astype(complex))
    cells = np.asarray(d.mesh.cells_dict["triangle"])
    cc = pts[cells].mean(axis=1)
    at_bump = (cc[:, 0] - 0.7) ** 2 + (cc[:, 1] - 0.3) ** 2 < 0.12**2

    assert est_smooth < 1e-9, f"a smooth real field must give ~0 estimate, got {est_smooth:.2e}"
    assert eta[at_bump].mean() > 5 * eta[~at_bump].mean(), "the Im-only feature must localise the indicator"
    assert est > 0.1, "the complex field's estimate must be nonzero (driven by Im)"


def test_indicator_is_byte_identical_on_a_real_field():
    """Backward compatibility: for a real field the modulus form ``|x|^2`` equals ``x^2``, so the
    complex-safe estimator returns exactly what the old real-only one did."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.06)
    pts = np.asarray(d.mesh.points)
    field = np.sin(3 * pts[:, 0]) * np.cos(2 * pts[:, 1])  # real, nonlinear
    eta_real, est_real = zz_error_indicators(d, field)
    eta_cplx, est_cplx = zz_error_indicators(d, field.astype(complex))  # imag == 0
    np.testing.assert_allclose(eta_real, eta_cplx, rtol=0, atol=1e-14)
    assert abs(est_real - est_cplx) < 1e-13


def test_complex_helmholtz_adaptive_solve_converges():
    """End-to-end: an absorbing complex Helmholtz with a localised source. The adaptive loop must
    return a complex field, grow the DOFs, and drive the error estimate monotonically down."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.07)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = (1 + 0.5j) * jno.np.exp(-((xi - 0.5) ** 2 + (yi - 0.5) ** 2) / (2 * 0.04**2))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - (100.0 + 2j) * (u * vi) - f * vi, u(xb, yb) - 0.0])
    assert fem.is_complex

    sol = np.asarray(fem.solve(adapt=jno.solve.remesh(theta=0.6, max_iters=4, refine_factor=1.7), solve_fn=sparse_lu_solve))
    assert np.iscomplexobj(sol) and float(np.abs(sol.imag).max()) > 1e-6, "must be a genuine complex solve"
    hist = fem.adapt_history
    assert len(hist) >= 3
    dofs = [h["n_dofs"] for h in hist]
    ests = [h["estimate"] for h in hist]
    assert dofs == sorted(dofs) and dofs[-1] > 2 * dofs[0], "the mesh must actually refine"
    assert ests[-1] < 0.5 * ests[0], f"the error estimate must fall as the mesh refines: {ests}"


def test_complex_adaptive_supports_scalar_p2():
    """A higher-order (P2) scalar complex field is supported: the estimator drives on the vertex
    DOFs (the first ``n_vertices`` entries). The loop runs and refines."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.09)
    u, phi = d.fem_symbols(order=2)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = (1 + 0.5j) * jno.np.exp(-((xi - 0.5) ** 2 + (yi - 0.5) ** 2) / (2 * 0.04**2))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - (80.0 + 2j) * (u * vi) - f * vi, u(xb, yb) - 0.0])
    sol = np.asarray(fem.solve(adapt=jno.solve.remesh(theta=0.6, max_iters=3, refine_factor=1.7), solve_fn=sparse_lu_solve))
    assert np.iscomplexobj(sol)
    assert fem.adapt_history[-1]["n_dofs"] > fem.adapt_history[0]["n_dofs"], "P2 complex adaptive must refine"


def test_adaptive_preserves_robin_source_across_remesh():
    """Regression: an absorbing/Robin BC is a **surface-integral** term. After a remesh the tagged
    boundary must re-derive on the new facets, or the source term references stale nodes and
    silently vanishes -- collapsing the solve to ``u = 0``. The adaptive loop re-materializes the
    coordinate-predicate tags each round, so the driven plane wave survives refinement.

    An absorbing box launched by a unit wave from the bottom has ``|u| ~ 1`` everywhere; a lost
    source gives ``~0`` -- a sharp, discriminating check."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.06)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    k = 2 * np.pi
    bx, by, _ = d.variable("bottom", where=lambda x, y: y < 1e-6)
    tx, ty, _ = d.variable("top", where=lambda x, y: y > 1 - 1e-6)
    lx, ly, _ = d.variable("left", where=lambda x, y: x < 1e-6)
    rx, ry, _ = d.variable("right", where=lambda x, y: x > 1 - 1e-6)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    ub, vb = u.bind(x=bx, y=by), phi.bind(x=bx, y=by)
    ut, vt = u.bind(x=tx, y=ty), phi.bind(x=tx, y=ty)
    ul, vl = u.bind(x=lx, y=ly), phi.bind(x=lx, y=ly)
    ur, vr = u.bind(x=rx, y=ry), phi.bind(x=rx, y=ry)
    fem = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y - k**2 * (u * vi),
            -(1j * k * ub - 2j * k) * vb,  # bottom: launch a unit upward wave + absorb
            -(1j * k * ut) * vt,
            -(1j * k * ul) * vl,
            -(1j * k * ur) * vr,  # top/left/right absorb (non-reflecting)
        ]
    )
    sol = np.asarray(fem.solve(adapt=jno.solve.remesh(theta=0.6, max_iters=3, refine_factor=1.6), solve_fn=sparse_lu_solve))
    hist = fem.adapt_history
    assert hist[-1]["n_dofs"] > hist[0]["n_dofs"], "the mesh must refine"
    assert float(np.abs(sol).mean()) > 0.3, f"the Robin source must survive remesh (|u| mean={np.abs(sol).mean():.3f})"


def test_absorbing_source_survives_repeated_remesh():
    """Regression for the stale-tag bug: on remesh, jNO must drop the OLD mesh's predicate-tag state
    so a re-tag re-derives the absorbing/source boundary cleanly. Otherwise stale surface-tag state
    corrupts the re-assembled Robin term and the driven field collapses after the first remesh --
    which silently breaks any field-parameter adaptive inverse-design loop that remeshes then rebuilds.

    A homogeneous absorbing box launched by a unit wave keeps ``|u| ~ O(1)`` across several remeshes;
    before the fix it collapsed (~0.25) after the first."""
    from jno.utils.solver.fem_adapt import remesh_with_mmg

    k = 2 * np.pi
    L = 3.0
    faces = {
        "bottom": lambda x, y: y < 1e-6,
        "top": lambda x, y: y > L - 1e-6,
        "left": lambda x, y: x < 1e-6,
        "right": lambda x, y: x > L - 1e-6,
    }

    def amp(d):
        u, phi = d.fem_symbols()
        xi, yi, _ = d.variable("interior", split=True)
        bx, by, _ = d.variable("bottom", split=True)
        tx, ty, _ = d.variable("top", split=True)
        lx, ly, _ = d.variable("left", split=True)
        rx, ry, _ = d.variable("right", split=True)
        ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
        ub, vb = u.bind(x=bx, y=by), phi.bind(x=bx, y=by)
        ut, vt = u.bind(x=tx, y=ty), phi.bind(x=tx, y=ty)
        ul, vl = u.bind(x=lx, y=ly), phi.bind(x=lx, y=ly)
        ur, vr = u.bind(x=rx, y=ry), phi.bind(x=rx, y=ry)
        fem = jno.fem(
            [
                ui.x * vi.x + ui.y * vi.y - k**2 * (u * vi),
                -(1j * k * ub - 2j * k) * vb,  # bottom: launch a unit wave + absorb
                -(1j * k * ut) * vt,
                -(1j * k * ul) * vl,
                -(1j * k * ur) * vr,  # non-reflecting frame
            ]
        )
        return float(np.abs(np.asarray(fem.solve(solve_fn=sparse_lu_solve))).mean())

    d = jno.domain(box(0, 0, L, L), mesh_size=0.14)
    for nm, pr in faces.items():
        d.tag(nm, pr)
    assert amp(d) > 0.5, "the driven absorbing box should have |u| ~ O(1) before remeshing"
    for _ in range(3):  # remesh (drops stale tag state) -> re-tag via predicates (the real-path pattern)
        remesh_with_mmg(d, np.full(np.asarray(d.built_mesh.points).shape[0], 0.11), copy=False)
        for nm, pr in list(getattr(d, "_tag_predicates", {}).items()):
            d.tag(nm, pr)
        a = amp(d)
        assert a > 0.5, f"the absorbing source must survive remesh (|u| mean {a:.3f}); stale-tag regression"


def test_adaptive_supports_vector_field():
    """A vector field now drives h-adaptivity: the ZZ estimator sums the per-component recovered-gradient
    errors (one scalar indicator per cell), so the adaptive solve runs and returns the vector solution
    rather than rejecting it. (Per-component recovery + estimate-drop are checked in tests/test_fem_adapt.py.)"""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.2)
    w, z = d.fem_symbols(value_shape=(2,), names=("w", "z"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    zi = z.bind(x=xi, y=yi)
    weak = jno.np.inner(jno.np.grad(w, [xi, yi]), jno.np.grad(z, [xi, yi]), n_contract=2) - (zi[0] + zi[1])
    fem = jno.fem([weak, w(xb, yb)[0] - 0.0, w(xb, yb)[1] - 0.0])
    sol = np.asarray(fem.solve(adapt=jno.solve.remesh(max_iters=2), solve_fn=sparse_lu_solve)).reshape(-1)
    assert sol.shape[0] == 2 * len(fem.domain.mesh.points), "vector adaptive solve returns 2 DOFs per node"


def test_adaptive_anisotropic_rejects_complex():
    """Anisotropic (Hessian-metric) adaptation is real-only; a complex field must raise, not run a
    meaningless complex-Hessian metric."""
    d = jno.domain(box(0, 0, 1, 1), mesh_size=0.12)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - (10.0 + 1j) * (u * vi) - (1 + 0.2j) * vi, u(xb, yb) - 0.0])
    with pytest.raises(NotImplementedError, match="real-only"):
        fem.solve(adapt=jno.solve.remesh(anisotropic=True, max_iters=2), solve_fn=sparse_lu_solve)


def test_adapt_complex_transient_decaying_mode():
    """``fem.solve(adapt=...)`` on a COMPLEX transient (previously fail-loud): the fused ``[Re; Im]``
    state transfers across each remesh as a doubled field layout, the MODULUS drives the metric, and
    the frames come back complex. Pinned against the analytic decaying mode of complex diffusion
    ``ψ_t = c Δψ`` (c = 0.5+1j, Dirichlet walls): ``ψ(t) = exp(-c·2π²·t)·sin(πx)sin(πy)``."""
    c = 0.5 + 1j
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.09, time=(0.0, 0.02, 41))
    u, phi = d.fem_symbols(names=("axc", "axcp"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    psi0 = jno.np.sin(np.pi * ci[0]) * jno.np.sin(np.pi * ci[1])
    fem = jno.fem([ui.t * vi + c * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - psi0])
    assert fem.is_complex and fem.is_transient
    traj = fem.solve(adapt=jno.solve.remesh(every=10, max_dofs=260))
    assert len(fem.adapt_history) >= 1, "the march must actually remesh (the transfer is the point)"

    state, (pts, _cells) = traj.final()
    state = np.asarray(state)
    assert np.iscomplexobj(state), "a complex transient's adaptive frames must be complex"
    pts = np.asarray(pts)
    assert state.shape[0] == pts.shape[0], "frame length must match its OWN mesh (n complex DOFs)"
    t1 = float(traj.times[-1])
    exact = np.exp(-c * 2 * np.pi**2 * t1) * np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
    rel = np.linalg.norm(state - exact) / np.linalg.norm(exact)
    assert rel < 6e-2, f"adapted complex transient vs analytic decaying mode: rel {rel:.3e}"
    assert float(np.abs(state.imag).max()) > 1e-2, "the decayed mode must be genuinely complex"

    # resample the complex trajectory onto the final mesh: complex frames, uniform array out
    frames = traj.resample(fem.domain)
    frames = np.asarray(frames)
    assert np.iscomplexobj(frames) and frames.shape[0] == len(traj)


def test_adapt_complex_transient_zero_ic_stays_zero():
    """Extreme: a zero IC must stay identically zero through every remesh — the doubled-layout
    transfer and the recombination must not invent a field on either half."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.12, time=(0.0, 0.02, 21))
    u, phi = d.fem_symbols(names=("zxc", "zxcp"))
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    fem = jno.fem([ui.t * vi + (0.5 + 1j) * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - 0.0])
    traj = fem.solve(adapt=jno.solve.remesh(every=8, max_dofs=200))
    for s in traj.states:
        assert float(np.abs(np.asarray(s)).max()) < 1e-12
