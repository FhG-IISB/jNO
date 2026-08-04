"""Structure-aware preconditioner specs: ``jno.precond.form`` (auxiliary weak-form operators),
``jno.precond.inner`` (a solver as the M⁻¹ application), and per-field block composition
(``block_diag`` / ``triangular``) over ``fem.blocks``.

Pins: the flagship saddle-point pattern — Taylor–Hood Stokes solved by FGMRES with an
upper-triangular block preconditioner (inexact CG velocity solve + (1/μ)-weighted pressure-mass
Schur approximation, Elman/Silvester/Wathen §9.2) against the sparse-direct reference, including
spec *reuse* across solves (cached auxiliary operator, snapshotted field keys); symbol→block
resolution in offsets order (`fem.blocks` / `fem.block_index`); sub-operator extraction
(``ctx.sub``) against dense slices; ``form`` on the full system; and the pair-validation guards.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _stokes(mesh_size=0.25):
    """Taylor-Hood (P2/P1) Poiseuille channel — the docs/tutorial Stokes system."""
    inner_, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    G, mu, H, Lx = 1.0, 1.0, 1.0, 4.0
    u_profile = lambda y: (G / (2 * mu)) * y * (H - y)
    d = jno.domain(box(0.0, 0.0, Lx, H), mesh_size=mesh_size)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            mu * inner_(gu, gv, n_contract=2) - pp * trace(gv),
            -qq * trace(gu),
            u(xb, yb)[0] - u_profile(yb),
            u(xb, yb)[1] - 0.0,
            p.pin(),
        ]
    )
    return fem, u, p, pp, qq, mu


def _poisson(mesh_size=0.2):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1.0 - xi) + yi * (1.0 - yi))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)
    return fem, ui, vi


# ---------------------------------------------------------------------------
# the flagship: Stokes saddle system, FGMRES + block preconditioning
# ---------------------------------------------------------------------------


def test_stokes_fgmres_triangular_schur():
    fem, u, p, pp, qq, mu = _stokes()
    u_ref = np.asarray(fem.solve(linear=jno.solve.lu()))
    tri = jno.precond.triangular(
        (u, jno.precond.inner(jno.solve.cg(tol=1e-2, maxiter=60))),  # inexact velocity block solve
        (p, jno.precond.form([(1.0 / mu) * pp * qq], inner=jno.solve.dense())),  # μ⁻¹-weighted pressure mass
    )
    sol = fem.solve(linear=jno.solve.fgmres(tol=1e-10, restart=40, maxiter=4000), precond=tri)
    assert np.abs(np.asarray(sol) - u_ref).max() < 1e-6
    # spec reuse: the auxiliary mass operator is cached, the field-key snapshot survives the
    # auxiliary assembly on the same domain (regression: domain._fem_native_field_keys overwrite)
    sol2 = fem.solve(linear=jno.solve.fgmres(tol=1e-10, restart=40, maxiter=4000), precond=tri)
    assert np.abs(np.asarray(sol2) - u_ref).max() < 1e-6


def test_points_survive_aux_assembly():
    """Regression: fem.points / fem.field_points are snapshots — assembling a jno.precond.form
    auxiliary operator on the same domain must not clobber them (the raw domain attributes
    _fem_native_dof_points* ARE overwritten by the aux assembly; the FEM must not care)."""
    fem, u, p, pp, qq, mu = _stokes(mesh_size=0.35)
    pts_before = [np.asarray(x) for x in fem.field_points]
    n_before = np.asarray(fem.points).shape[0]
    fem.solve(
        linear=jno.solve.fgmres(tol=1e-8, restart=40, maxiter=4000),
        precond=jno.precond.triangular(
            (u, jno.precond.inner(jno.solve.cg(tol=1e-2, maxiter=60))),
            (p, jno.precond.form([(1.0 / mu) * pp * qq], inner=jno.solve.dense())),
        ),
    )
    pts_after = fem.field_points
    assert len(pts_after) == len(pts_before)
    for a, b in zip(pts_after, pts_before):
        assert np.asarray(a).shape == b.shape and np.allclose(np.asarray(a), b)
    assert np.asarray(fem.points).shape[0] == n_before


def test_stokes_block_diag():
    fem, u, p, pp, qq, mu = _stokes()
    u_ref = np.asarray(fem.solve(linear=jno.solve.lu()))
    bd = jno.precond.block_diag(
        (u, jno.precond.inner(jno.solve.cg(tol=1e-2, maxiter=60))),
        (p, jno.precond.form([(1.0 / mu) * pp * qq], inner=jno.solve.dense())),
    )
    sol = fem.solve(linear=jno.solve.fgmres(tol=1e-10, restart=40, maxiter=4000), precond=bd)
    assert np.abs(np.asarray(sol) - u_ref).max() < 1e-6


# ---------------------------------------------------------------------------
# structure handles
# ---------------------------------------------------------------------------


def test_blocks_and_block_index_offsets_order():
    fem, u, p, *_ = _stokes(mesh_size=0.35)
    off = fem.offsets
    assert fem.blocks == [slice(int(off[0]), int(off[1])), slice(int(off[1]), int(off[2]))]
    iu, ip = fem.block_index(u), fem.block_index(p)
    assert (iu, ip) == (0, 1)  # P2 vector velocity is the big first block
    assert (off[1] - off[0]) > (off[2] - off[1])
    assert fem.block_index(1) == 1  # integer passthrough
    # a foreign symbol is rejected
    d2 = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.5)
    w, _ = d2.fem_symbols(names=("w", "pw"))
    with pytest.raises(KeyError, match="not part of this system"):
        fem.block_index(w)


def test_single_field_blocks_trivial():
    """A single-field system has no *useful* block structure: blocks is None (no offsets) or the
    one whole-vector slice (native path sets offsets [0, N])."""
    fem, *_ = _poisson()
    assert fem.blocks is None or fem.blocks == [slice(0, fem.dofs)]


def test_sub_operator_matches_dense_slices():
    from jno.utils.solver.solver_api import LinearOperator, PrecondContext

    fem, u, p, *_ = _stokes(mesh_size=0.35)
    ctx = PrecondContext(LinearOperator(fem.A), fem)
    Ad = np.asarray(fem.A.todense() if hasattr(fem.A, "todense") else fem.A)
    su, sp = fem.blocks
    Auu, Aup = Ad[su, su], Ad[su, sp]
    vu = jnp.asarray(np.random.default_rng(0).standard_normal(Auu.shape[0]))
    vp = jnp.asarray(np.random.default_rng(1).standard_normal(Aup.shape[1]))
    assert np.allclose(np.asarray(ctx.sub(u).mv(vu)), Auu @ np.asarray(vu))
    assert np.allclose(np.asarray(ctx.sub(u, p).mv(vp)), Aup @ np.asarray(vp))
    assert np.allclose(np.asarray(ctx.sub(u, p).T.mv(vu)), Aup.T @ np.asarray(vu))
    assert np.allclose(np.asarray(ctx.sub(u).diag()), np.diag(Auu))
    assert np.allclose(np.asarray(ctx.sub(u, p).dense()), Aup)
    assert ctx.sub(u, p).shape == Aup.shape


# ---------------------------------------------------------------------------
# form / inner on the full system
# ---------------------------------------------------------------------------


def test_form_full_system_preconditioner():
    """The operator's own weak form as the aux operator == an (up to BC rows) exact
    preconditioner; any Krylov converges immediately to the reference."""
    fem, ui, vi = _poisson()
    u_ref = np.asarray(fem.solve())
    spec = jno.precond.form([ui.x * vi.x + ui.y * vi.y + ui * vi], inner=jno.solve.lu(), quad_degree=3)
    sol = fem.solve(linear=jno.solve.fgmres(tol=1e-10), precond=spec)
    assert np.abs(np.asarray(sol) - u_ref).max() < 1e-6
    # materialization is cached: the second materialize returns without re-assembling
    from jno.utils.solver.solver_api import LinearOperator, PrecondContext

    ctx = PrecondContext(LinearOperator(fem.A), fem)
    spec.materialize(ctx)
    op_first = spec._op
    spec.materialize(ctx)
    assert spec._op is op_first


def test_form_precond_accepts_complex_shifted_laplacian():
    """A COMPLEX auxiliary operator -- the shifted-Laplacian twin of a complex Helmholtz -- now
    assembles as the matching ``2n`` real-equivalent block ``[[Mr,-Mi],[Mi,Mr]]`` and preconditions
    the complex solve. On an indefinite Helmholtz, ``gmres + form(shifted)`` converges to the
    sparse-direct reference (before the fix this raised "must be steady linear")."""
    from jno.utils.solver.linear import sparse_lu_solve

    k = 2 * np.pi
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.05)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    src = jno.np.exp(-(((xi - 0.5) ** 2 + (yi - 0.5) ** 2) / (2 * 0.1**2)))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y - (k**2 + 2j) * (u * vi) - src * vi, u(xb, yb) - 0.0])
    assert fem.is_complex
    u_ref = np.asarray(fem.solve(solve_fn=sparse_lu_solve))

    shifted = [ui.x * vi.x + ui.y * vi.y - (k**2 + 1j * 0.5 * k**2) * (u * vi), u(xb, yb) - 0.0]  # damped twin
    sol = np.asarray(
        fem.solve(
            linear=jno.solve.gmres(tol=1e-10, maxiter=500, restart=60),
            precond=jno.precond.form(shifted, inner=jno.solve.lu()),
        )
    )
    assert np.iscomplexobj(sol)
    rel = float(np.linalg.norm(sol - u_ref) / np.linalg.norm(u_ref))
    assert rel < 1e-6, f"complex shifted-Laplacian preconditioned solve must match the direct reference: {rel:.2e}"


def test_form_precond_still_rejects_nonlinear_aux():
    """The steady-linear guard still holds: a nonlinear auxiliary form is rejected (the complex
    branch widened the guard to steady-linear real *or* complex, not to nonlinear/transient)."""
    fem, ui, vi = _poisson()
    with pytest.raises(ValueError, match="steady linear"):
        fem.solve(linear=jno.solve.fgmres(), precond=jno.precond.form([ui.x * vi.x + (ui * ui) * vi]))


def test_inner_spec_full_system():
    fem, *_ = _poisson()
    u_ref = np.asarray(fem.solve())
    sol = fem.solve(
        linear=jno.solve.fgmres(tol=1e-10),
        precond=jno.precond.inner(jno.solve.cg(tol=1e-1, maxiter=25)),
    )
    assert np.abs(np.asarray(sol) - u_ref).max() < 1e-6


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------


def test_block_pair_validation():
    fem, u, p, pp, qq, mu = _stokes(mesh_size=0.35)
    jac = jno.precond.jacobi()
    with pytest.raises(ValueError, match="specified twice"):
        fem.solve(linear=jno.solve.fgmres(), precond=jno.precond.triangular((u, jac), (u, jac)))
    with pytest.raises(ValueError, match="every field needs exactly one"):
        fem.solve(linear=jno.solve.fgmres(), precond=jno.precond.triangular((u, jac)))


# --------------------------------------------------------------------------------------------
# a block preconditioner's DIAGONAL blocks must be assembled, not matvec-only
# --------------------------------------------------------------------------------------------
def _two_field_stokes(ms=0.25):
    """A small Taylor-Hood Stokes system and its (u, p) trial symbols."""
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 2.0, 1.0), mesh_size=ms)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    gu, gv = jno.np.grad(u, [xi, yi]), jno.np.grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            jno.np.inner(gu, gv, n_contract=2) - pp * jno.np.trace(gv),
            -qq * jno.np.trace(gu),
            u(xb, yb)[0] - yb * (1.0 - yb),
            u(xb, yb)[1] - 0.0,
            p.pin(),
        ]
    )
    return fem, u, p, pp, qq


def test_diagonal_block_is_assembled_not_matvec_only():
    """The block a preconditioner SOLVES must carry a sparse matrix.

    Handing it back matvec-only silently restricted the inner solver to matrix-free methods, so
    `inner(jno.solve.amg())` / `inner(jno.solve.lu())` were refused -- which rules out the standard
    saddle-point recipe (multigrid on the velocity block).
    """
    from jno.utils.solver.solver_api import PrecondContext

    fem, u, p, _pp, _qq = _two_field_stokes()
    ctx = PrecondContext(_op(fem), fem=fem)
    for i in (0, 1):  # block 0 velocity, block 1 pressure
        blk = ctx.sub(i)
        assert blk.bcoo is not None, f"diagonal block {i} has no assembled matrix"
        assert blk.shape[0] == blk.shape[1]


def _op(fem):
    from jno.utils.solver.solver_api import LinearOperator

    A, _b = fem.operator
    return LinearOperator(A)


def test_diagonal_block_matches_the_dense_sub_matrix():
    """The assembled block must equal the corresponding sub-matrix of the parent, exactly."""
    from jno.utils.solver.solver_api import PrecondContext

    fem, _u, _p, _pp, _qq = _two_field_stokes()
    A, _b = fem.operator
    ctx = PrecondContext(_op(fem), fem=fem)
    full = np.asarray(A.todense())
    for i in (0, 1):
        s = ctx.block_slice(i)
        got = np.asarray(ctx.sub(i).bcoo.todense())
        assert np.allclose(got, full[s, s], atol=1e-12), f"block {i} is not the parent's sub-matrix"


def test_offdiagonal_block_still_applies_correctly():
    """Off-diagonal blocks are only ever APPLIED, so they may stay matvec-only -- but must be right."""
    from jno.utils.solver.solver_api import PrecondContext

    fem, _u, _p, _pp, _qq = _two_field_stokes()
    A, _b = fem.operator
    ctx = PrecondContext(_op(fem), fem=fem)
    full = np.asarray(A.todense())
    si, sj = ctx.block_slice(0), ctx.block_slice(1)
    v = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (sj.stop - sj.start,)), dtype=full.dtype)
    got = np.asarray(ctx.sub(0, 1).mv(jnp.asarray(v)))
    assert np.allclose(got, full[si, sj] @ v, atol=1e-10)


def test_block_diagonal_solve_with_a_sparse_inner_solver():
    """End to end: a block spec whose velocity block needs an ASSEMBLED operator now runs."""
    fem, u, p, pp, qq = _two_field_stokes()
    A, b = fem.operator
    precond = jno.precond.triangular(
        (u, jno.precond.inner(jno.solve.lu())),  # needs the assembled block; used to be refused
        (p, jno.precond.form([pp * qq])),
    )
    x = np.asarray(fem.solve(linear=jno.solve.fgmres(), precond=precond)).reshape(-1)
    rel = np.linalg.norm(np.asarray(A @ x) - np.asarray(b).reshape(-1)) / np.linalg.norm(np.asarray(b))
    assert rel < 1e-6, f"block-preconditioned Stokes did not converge: {rel:.2e}"
