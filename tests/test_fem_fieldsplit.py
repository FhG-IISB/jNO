"""Fieldsplit completion — the three gaps that stalled the rigid-plastic rolling model.

That project needed (1) a Schur-complement approximation whose weight depends on the SOLUTION
(``(1/mu(u))``-weighted pressure mass), (2) a real (sparse) AMG on a velocity sub-block, and (3) any
visibility into what the solver did. ``precond.form`` assembled once and was parameter-independent,
``ctx.sub`` handed sub-blocks back dense-or-matvec, and nothing reported anything.

Now: ``form(<callable>)`` re-assembles from the outer iterate (the Picard-lagged preconditioner —
Elman, Silvester & Wathen, *Finite Elements and Fast Iterative Solvers*, 2nd ed., §9.2), diagonal
sub-blocks come back as assembled BCOO, and ``fem.stats`` reports the drivers' outcome.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno


def _nonlinear_diffusion(size=0.2):
    """-div((1+u^2) grad u) = f — the smallest problem with a genuinely solution-dependent operator."""
    d = jno.Shape.rect(0, 0, 1, 1, size=size).domain()
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    fem = jno.fem([(1.0 + ui * ui) * (ui.x * vi.x + ui.y * vi.y) - 5.0 * vi, u(xb, yb) - 0.0])
    return fem, ui, vi


def _stokes(mesh_size=0.3):
    """Taylor-Hood Poiseuille channel — the saddle system block preconditioners exist for."""
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

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


# --------------------------------------------------------------------------------------
# solution-dependent precond.form — the Picard-lagged Schur weight
# --------------------------------------------------------------------------------------
def test_solution_dependent_form_solves_and_matches():
    """The rigid-plastic spelling: a form whose coefficient is computed FROM the solution. It refreshes
    from the outer iterate and the preconditioned solve must land on the plain solve's answer —
    a preconditioner changes speed, never the solution."""
    fem, ui, vi = _nonlinear_diffusion()
    ref = fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-6, atol=1e-6), linear=jno.solve.lu(backend="host"))

    M = jno.precond.form(
        lambda sol: [(1.0 + float(np.mean(np.asarray(sol) ** 2))) * ui * vi],
        inner=jno.solve.lu(backend="host"),
    )
    got = fem.solve(nonlinear=jno.solve.newton(rtol=1e-6, atol=1e-6), linear=jno.solve.fgmres(tol=1e-8), precond=M)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-3, atol=1e-4)
    assert M.terms is not None, "the callable form was never refreshed from the iterate"


def test_solution_dependent_form_refreshes_per_solve():
    """Two solves = two refreshes, each from that solve's entry iterate (the lag is per OUTER solve)."""
    fem, ui, vi = _nonlinear_diffusion()
    seen = []

    def terms_fn(sol):
        seen.append(float(np.linalg.norm(sol)))
        return [(1.0 + float(np.mean(sol**2))) * ui * vi]

    M = jno.precond.form(terms_fn, inner=jno.solve.lu(backend="host"))
    kw = dict(nonlinear=jno.solve.newton(rtol=1e-6, atol=1e-6), linear=jno.solve.fgmres(tol=1e-8), precond=M)
    s1 = fem.solve(**kw)
    fem.solve(x0=s1, **kw)
    assert len(seen) == 2, f"expected one refresh per solve, saw {len(seen)}"
    assert seen[0] != seen[1], "the second refresh should see the first solve's answer as its iterate"


def test_solution_dependent_form_never_assembled_raises_loudly():
    """No concrete iterate -> the spec must refuse at materialization, naming the path — not
    silently precondition with garbage."""
    from jno.utils.solver.solver_api import LinearOperator, PrecondContext

    fem, ui, vi = _nonlinear_diffusion()
    M = jno.precond.form(lambda sol: [ui * vi])
    with pytest.raises(NotImplementedError, match="never assembled"):
        M.materialize(PrecondContext(LinearOperator.from_matvec(lambda v: v, shape=(4, 4)), fem))


def test_static_form_is_unchanged():
    """The list form keeps its one-time parameter-independent assembly — no behavior change."""
    fem, ui, vi = _nonlinear_diffusion()
    M = jno.precond.form([ui * vi], inner=jno.solve.lu(backend="host"))
    ref = fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-6, atol=1e-6), linear=jno.solve.lu(backend="host"))
    got = fem.solve(nonlinear=jno.solve.newton(rtol=1e-6, atol=1e-6), linear=jno.solve.fgmres(tol=1e-8), precond=M)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=1e-3, atol=1e-4)


# --------------------------------------------------------------------------------------
# sparse sub-blocks — AMG on the velocity block of a saddle system
# --------------------------------------------------------------------------------------
def test_diagonal_sub_block_is_assembled_sparse():
    """`ctx.sub(i, i)` must hand back an assembled sub-operator (`.bcoo` or dense), not a
    matvec-only view — that is what lets `inner(lu)` / `amg` run on a velocity block at all."""
    from jno.utils.solver.solver_api import PrecondContext

    fem, u, p, *_ = _stokes()
    _ = np.asarray(fem.b)
    from jno.utils.solver.solver_api import LinearOperator

    ctx = PrecondContext(LinearOperator(fem.A), fem)
    sub = ctx.sub(fem.block_index(u))
    assert sub.bcoo is not None or sub.dense() is not None
    nu = fem.blocks[fem.block_index(u)]
    assert sub.shape == (nu.stop - nu.start, nu.stop - nu.start)
    # and it is the REAL block: its diagonal equals the parent's slice
    np.testing.assert_allclose(np.asarray(sub.diag()), np.asarray(LinearOperator(fem.A).diag()[nu]), rtol=1e-6, atol=1e-7)


def test_stokes_triangular_with_amg_velocity_block():
    """The full rigid-plastic-shaped recipe on a saddle system: triangular fieldsplit, multigrid on the
    velocity sub-block, weighted pressure mass as the Schur approximation."""
    pytest.importorskip("pyamg", reason="pyamg required for the hybrid AMG spec")
    fem, u, p, pp, qq, mu = _stokes()
    # The DEFAULT (Jacobi-BiCGStab) is exactly what fails on an indefinite saddle -- that is why
    # fieldsplit exists. The oracle is a direct solve.
    ref = fem.solve(linear=jno.solve.lu(backend="host"))

    tri = jno.precond.triangular(
        (u, jno.precond.amg()),
        (p, jno.precond.form([(1.0 / mu) * pp * qq], inner=jno.solve.lu(backend="host"))),
    )
    got = fem.solve(linear=jno.solve.fgmres(tol=1e-9), precond=tri)
    # float32 Krylov floor: measured rel err 6.3e-4 against the direct solve at these settings.
    err = np.linalg.norm(np.asarray(got) - np.asarray(ref)) / np.linalg.norm(np.asarray(ref))
    assert err < 5e-3, f"fieldsplit solve drifted from the direct oracle: rel err {err:.2e}"
    assert fem.stats is not None and "triangular" in str(fem.stats["precond"])


# --------------------------------------------------------------------------------------
# fem.stats — observability
# --------------------------------------------------------------------------------------
def test_stats_reports_the_newton_outcome():
    fem, *_ = _nonlinear_diffusion()
    fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-6, atol=1e-6), linear=jno.solve.lu(backend="host"))
    st = fem.stats
    assert st["mode"] == "nonlinear" and st["dofs"] > 0
    nl = st["nonlinear"]
    assert nl["driver"] == "newton_direct" and nl["converged"] and nl["steps"] >= 1
    assert nl["residual"] <= nl["bound"]


def test_stats_none_before_any_solve():
    fem, *_ = _nonlinear_diffusion()
    assert fem.stats is None


def test_stats_records_the_slots():
    fem, u, p, pp, qq, mu = _stokes(mesh_size=0.4)
    fem.solve(
        linear=jno.solve.fgmres(tol=1e-9),
        precond=jno.precond.block_diag(
            (u, jno.precond.inner(jno.solve.cg(tol=1e-4))),
            (p, jno.precond.form([(1.0 / mu) * pp * qq], inner=jno.solve.lu(backend="host"))),
        ),
    )
    st = fem.stats
    assert "fgmres" in st["linear"] and "block_diag" in st["precond"]


# --------------------------------------------------------------------------------------
# refresh cadence on cached specs
# --------------------------------------------------------------------------------------
def test_cached_refresh_int_rebuilds_on_cadence():
    """`cached(spec, refresh=k)`: reuse within the window, rebuild on the k-th materialization."""
    import jax.numpy as jnp

    from jno.precond import PrecondContext, cached, jacobi
    from jno.utils.solver.solver_api import LinearOperator

    op = LinearOperator.from_matvec(lambda v: v, diag_fn=lambda: jnp.ones(4), shape=(4, 4))
    ctx = PrecondContext(op)
    c = cached(jacobi(), refresh=2)
    a1, a2, a3, a4, a5 = (c.materialize(ctx) for _ in range(5))
    assert a1 is a2, "within the k-window the applier must be reused"
    assert a3 is not a2, "the k-th materialization must rebuild"
    assert a3 is a4 and a5 is not a4


def test_cached_refresh_rejects_nonsense():
    from jno.precond import cached, jacobi

    c = cached(jacobi(), refresh="often")
    from jno.precond import PrecondContext
    from jno.utils.solver.solver_api import LinearOperator

    with pytest.raises(TypeError, match="expected bool, int, or callable"):
        c.materialize(PrecondContext(LinearOperator.from_matvec(lambda v: v, shape=(2, 2))))
