"""Complex-valued FEM through the real-equivalent block (real-only assembly).

``jno.fem`` detects a complex weak form, splits each term into real Re/Im sub-forms
(``Re(c·T)=Re(c)·T`` since the FE trial/test ``T`` is real), assembles both through the ordinary
**real** assembly path, solves the real block ``[[A_r,-A_i],[A_i,A_r]]``, and returns ``u_r + i·u_i``.
The complex problem is handled entirely by the two coupled real systems.

Run with x64 (the solution is complex128): ``JAX_ENABLE_X64=1``.
"""

import pytest

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
    assert fem.problem is None  # the Re/Im real systems are assembled natively

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
    assert fem.problem is None  # native real-equivalent assembly (Dirichlet wall included)

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
    assert fem.problem is None  # M_r/A_r and M_i=0/A_i blocks are assembled natively
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


def _manufactured_complex_fem(mesh_size=0.1):
    """The all-Neumann manufactured complex Helmholtz of the first test, as a reusable fixture."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    c, d_coef, amp = 1.0 / (1.0 + 1j * 0.5), -(1.0 + 0.2j), 1.0 + 0.5j
    g = jno.np.cos(PI * xi) * jno.np.cos(PI * yi)
    f = (2 * PI**2 * c + d_coef) * amp * g
    return jno.fem([c * (ui.x * vi.x + ui.y * vi.y) + d_coef * (u * vi) - f * vi])


def test_steady_complex_is_one_fused_real_block():
    """A steady complex form is assembled as ONE real ``2n`` system ``[[A_r,-A_i],[A_i,A_r]]`` over
    ``x=[x_r; x_i]``, not as a pair of legs welded together at solve time — so it is an ordinary
    ``"linear"`` FEM and every real-linear facility applies to it unchanged. The unfused legs stay
    reachable for a complex-native preconditioner (AMS), which wants ``A_r + i·A_i`` instead."""
    fem = _manufactured_complex_fem()
    n = int(np.asarray(fem.points).shape[0])
    assert fem.is_complex and fem._mode == "linear", "a fused complex problem is an ordinary linear one"
    assert fem._complex_n == n
    A, b = fem._op
    assert A.shape == (2 * n, 2 * n) and b.shape == (2 * n,), f"expected the 2n block, got {A.shape}"
    assert hasattr(A, "indices"), "the fused block must stay sparse (BCOO)"
    assert not np.iscomplexobj(np.asarray(A.data)), "the real-equivalent block must be real"
    assert fem._complex_legs is not None and len(fem._complex_legs) == 2
    # and it still returns an n-vector complex solution, not the 2n internal layout
    u = np.asarray(fem.solve())
    assert u.shape == (n,) and np.iscomplexobj(u)


def test_fused_block_and_the_retained_legs_stay_in_sync():
    """The two representations of a complex problem must keep describing the SAME operator.

    ``.operator`` is the fused real ``2n`` block; ``_complex_legs`` keeps the unfused ``(re, im)``
    pair for the consumers that still need it (``jno.rcwa``'s source reader, the complex-native AMS
    preconditioner). Nothing structural forces the two to agree.

    This is pinned because a stale reader of the legs does NOT crash: the fused value is *also* a
    2-tuple, so ``op_r, op_i = fem.operator`` keeps unpacking cleanly and silently binds ``op_i`` to
    the LOAD VECTOR. That is exactly how it failed once — ``jno.rcwa`` lost its front door, and a
    Nedelec test reported a vanished imaginary part that was present the whole time. A future change
    to either layout should break this one clearly-named test, with the consumer list sitting next to
    ``_complex_legs`` in ``_fem.py``, rather than surfacing far away as a wrong number."""
    from jno._fem import _complex_block_bcoo

    fem = _manufactured_complex_fem()
    n = int(fem._complex_n)
    (A_r, b_r), (A_i, b_i) = fem._complex_legs
    A_fused, b_fused = fem.operator
    dense = lambda A: np.asarray(A.todense() if hasattr(A, "todense") else A)  # noqa: E731

    ref = dense(_complex_block_bcoo(A_r, A_i, n))
    got = dense(A_fused)
    scale = max(1.0, float(np.max(np.abs(ref))))
    assert np.max(np.abs(got - ref)) < 1e-12 * scale, "the fused block is no longer [[A_r,-A_i],[A_i,A_r]]"

    rhs = np.concatenate([np.asarray(b_r).reshape(-1), np.asarray(b_i).reshape(-1)])
    assert np.max(np.abs(np.asarray(b_fused).reshape(-1) - rhs)) < 1e-12 * max(1.0, float(np.max(np.abs(rhs))))


def test_steady_complex_accepts_a_warm_start():
    """``x0=`` on a complex problem was refused outright ("the solve runs on the real-equivalent block,
    an internal layout"). With the block built at assembly the guess is just a vector in that layout, so
    a **complex** x0 is accepted and mapped to ``[Re; Im]`` — and warm-starting from the answer must
    reproduce the answer."""
    fem = _manufactured_complex_fem()
    u_ref = np.asarray(fem.solve())

    warm = np.asarray(_manufactured_complex_fem().solve(linear=jno.solve.gmres(tol=1e-12), x0=u_ref))
    assert np.iscomplexobj(warm) and warm.shape == u_ref.shape
    assert float(np.linalg.norm(warm - u_ref) / np.linalg.norm(u_ref)) < 1e-8, "warm start changed the solution"

    # a cold zero guess reaches the same solution (x0 must be a guess, never a constraint)
    cold = np.asarray(_manufactured_complex_fem().solve(linear=jno.solve.gmres(tol=1e-12), x0=np.zeros_like(u_ref)))
    assert float(np.linalg.norm(cold - u_ref) / np.linalg.norm(u_ref)) < 1e-6


def test_complex_transient_save_ts_interpolates_every_frame():
    """CHARACTERIZATION — the ``save_ts`` contract on a complex transient, pinned frame by frame.

    The existing complex-transient tests check only ``traj[-1]``, so a change that got the endpoint
    right while corrupting the interior of the march (or the ``save_ts`` interpolation onto times that
    are *not* natural grid points) would pass unnoticed. Here the requested times are deliberately the
    grid **midpoints**, which forces the interpolation path, and *every* returned frame is compared
    against the analytic mode

        psi(t) = exp(-c 2 pi^2 t) sin(pi x) sin(pi y),   c = 0.5 + 1j.

    Pins the shape contract (one frame per requested time), the time placement, and the per-frame
    accuracy — all behaviour, no internal representation, so it holds across the assembly refactor."""
    nsteps, t_end = 41, 0.04
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.07, time=(0.0, t_end, nsteps))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    c = 0.5 + 1j
    psi0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem([ui.t * vi + c * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - psi0])

    grid = np.linspace(0.0, t_end, nsteps)
    save = 0.5 * (grid[:-1] + grid[1:])  # midpoints: never a natural step, so interpolation must fire
    traj = np.asarray(fem.solve(save_ts=save))
    pts = np.asarray(fem.points)

    assert traj.shape[0] == save.shape[0], f"expected one frame per save_ts, got {traj.shape[0]} for {save.shape[0]}"
    assert np.iscomplexobj(traj) and not bool(np.isnan(traj).any())

    mode = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    errs = [
        float(
            np.linalg.norm(traj[i] - np.exp(-c * 2 * PI**2 * t) * mode) / np.linalg.norm(np.exp(-c * 2 * PI**2 * t) * mode)
        )
        for i, t in enumerate(save)
    ]
    # measured max is 8.6e-3 (backward-Euler error, largest at the final frame); 1.5e-2 leaves ~1.7x
    # headroom — tight enough that a mis-sized step or a broken interpolation trips it
    assert max(errs) < 1.5e-2, f"per-frame complex-transient error too large: max {max(errs):.3e} over {len(errs)} frames"
    # the decay must be monotone in |psi| — catches a march that lands the endpoint but wanders between
    norms = np.linalg.norm(traj, axis=1)
    assert np.all(np.diff(norms) < 0), "|psi| must decay monotonically for a damped complex diffusion"


def test_complex_transient_block_is_sparse_not_dense():
    """The complex-transient real-equivalent block ``[[·,-·],[·,·]]`` is composed as **one** BCOO at
    assembly and marched matrix-free (GMRES + Jacobi), so the dense ``(2N × 2N)`` block and its dense LU
    never materialise — a large complex diffusion / Schrödinger solve scales instead of hitting the
    ``O((2N)^2)`` memory wall. Assert the fused block is sparse and square at ``2n`` (a dense assembler
    would give plain ndarrays, with no ``.indices``), and that the sparse march recovers the analytic mode."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.06, time=(0.0, 0.03, 21))
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ci = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    psi0 = jno.np.sin(PI * ci[0]) * jno.np.sin(PI * ci[1])
    fem = jno.fem([ui.t * vi + (0.5 + 1j) * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(ci[0], ci[1]) - psi0])
    assert fem.is_complex and fem.is_transient
    block = fem.operator  # ONE real 2n SemidiscreteTimeBlock over the stacked [Re; Im] state
    # the fused block is BCOO (a dense global jacfwd / _as_dense would give ndarrays, no `.indices`)
    assert hasattr(block.M, "indices") and hasattr(block.A, "indices"), "the fused complex block must be sparse"
    n_pts = int(np.asarray(fem.points).shape[0])
    assert block.A.shape == (2 * n_pts, 2 * n_pts), f"expected the 2n real-equivalent block, got {block.A.shape}"
    assert block.M.shape == block.A.shape
    # the block is a real-equivalent embedding, so it must carry no complex dtype itself
    assert not np.iscomplexobj(np.asarray(block.A.data)), "the real-equivalent block must be real"
    # the sparse GMRES march still recovers exp(-c·2π²t)·sin·sin (accuracy is not sacrificed for sparsity)
    traj, pts = np.asarray(fem.solve()), np.asarray(fem.points)
    mode = np.sin(PI * pts[:, 0]) * np.sin(PI * pts[:, 1])
    analytic = np.exp(-(0.5 + 1j) * 2 * PI**2 * float(fem.t1)) * mode
    rel = float(np.linalg.norm(traj[-1] - analytic) / np.linalg.norm(analytic))
    assert rel < 3e-2, f"sparse complex-transient march rel-L2 {rel:.3e}"
    assert np.iscomplexobj(traj) and float(np.abs(traj[-1].imag).max()) > 1e-2  # genuinely complex


# ==========================================================================
# essential values on a complex form
# ==========================================================================
def test_real_dirichlet_on_a_complex_form_pins_re_and_zeroes_im():
    """A **real** essential value is well posed on a complex form: the two Re/Im legs share one
    Dirichlet row set, so the fused block imposes ``x_r - x_i = g`` and ``x_r + x_i = g``, i.e.
    ``Re u = g`` with ``Im u = 0`` on the region. The interior stays genuinely complex.

    ``-lap u + i u = 0`` on the unit square with ``u = x`` on the boundary."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y + 1j * (ui * vi), u(xb, yb) - xb])
    assert fem.is_complex

    uh = np.asarray(fem.solve()).reshape(-1)
    pts = np.asarray(fem.points)
    on_b = (
        (np.abs(pts[:, 0]) < 1e-9)
        | (np.abs(pts[:, 0] - 1.0) < 1e-9)
        | (np.abs(pts[:, 1]) < 1e-9)
        | (np.abs(pts[:, 1] - 1.0) < 1e-9)
    )
    assert np.max(np.abs(uh[on_b].real - pts[on_b, 0])) < 1e-12, "Re u = g not imposed"
    assert np.max(np.abs(uh[on_b].imag)) < 1e-12, "Im u must be pinned to zero by a real value"
    assert np.max(np.abs(uh[~on_b].imag)) > 1e-3, "the i*u reaction must make the interior complex"


def test_complex_dirichlet_value_fails_loud():
    """A **complex** essential value is NOT expressible on the shared Dirichlet row set: pinning
    ``Im u = g_i`` needs the imaginary leg's rows zeroed rather than set to identity, and the symmetric
    elimination's known-column lift is cross-leg (the real equation needs ``A_r[:,j] g_r - A_i[:,j] g_i``,
    which no per-leg elimination produces).

    This used to be *silent*: the value was cast with ``.astype(float)``, so ``Im(g)`` vanished behind a
    numpy ComplexWarning and the solve returned a plausible, wrong field — measured 8.9e-1 relative
    error on ``u = (1+2j)x``. It must refuse instead."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    with pytest.raises(NotImplementedError, match="COMPLEX essential value"):
        jno.fem([ui.x * vi.x + ui.y * vi.y + 0.0j * (ui * vi), u(xb, yb) - (1.0 + 2.0j) * xb])


def test_complex_constant_dirichlet_value_fails_loud():
    """The constant-profile branch resolves ``g`` by a separate single-point evaluation, so it needs the
    same guard — a constant ``1+2j`` must not slip through where a coordinate profile is refused."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    with pytest.raises(NotImplementedError, match="COMPLEX essential value"):
        jno.fem([ui.x * vi.x + ui.y * vi.y + 0.0j * (ui * vi), u(xb, yb) - (1.0 + 2.0j)])
