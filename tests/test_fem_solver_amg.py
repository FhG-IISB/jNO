"""The hybrid AMG preconditioner ``jno.precond.amg``: pyamg smoothed-aggregation setup on the
host (Vaněk/Mandel/Brezina 1996; PyAMG, Bell et al. 2023), pure-JAX Chebyshev-smoothed V-cycle
apply (Adams et al. 2003).

Pins: CG+AMG matches the sparse-direct reference on a Poisson system large enough for a real
multilevel hierarchy; the applier is ``jit``- and ``vmap``-native after an eager ``build`` (the
batch shares one frozen hierarchy) and is exactly *linear* (CG-legality); the saddle-point
architecture — FGMRES + triangular(velocity→AMG, pressure→weighted mass) — solves Taylor–Hood
Stokes; setup under a trace raises with ``.build()`` guidance; and pyamg stays **optional**
(clear ImportError, ``jno.precond`` imports fine without it)."""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("shapely", reason="shapely required for the box domain")
from shapely.geometry import box  # noqa: E402

import jno  # noqa: E402

pyamg = pytest.importorskip("pyamg", reason="pyamg required for the AMG setup (optional dep)")


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _poisson(mesh_size=0.03):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))
    return jno.fem([ui.x * vi.x + ui.y * vi.y - f * vi, u(xb, yb) - 0.0], quad_degree=3)


def test_cg_amg_matches_direct_on_poisson():
    fem = _poisson()
    assert fem.dofs > 1000  # large enough for a genuine multilevel hierarchy
    u_ref = np.asarray(fem.solve(linear=jno.solve.lu()))
    u = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-10), precond=jno.precond.amg()))
    assert np.abs(u - u_ref).max() < 1e-8


def test_amg_prebuilt_jit_vmap_and_linearity():
    from jno.utils.solver.solver_api import LinearOperator, PrecondContext

    fem = _poisson()
    u_ref = np.asarray(fem.solve(linear=jno.solve.lu()))
    spec = jno.precond.amg().build(fem.A)  # eager setup; frozen hierarchy
    op = LinearOperator(fem.A)
    M = spec.materialize(PrecondContext(op, fem))
    solver = jno.solve.cg(tol=1e-10)
    b = jnp.asarray(fem.b).reshape(-1)
    x = jax.jit(lambda bb: solver(op, bb, M=M))(b)
    assert np.abs(np.asarray(x) - u_ref).max() < 1e-8
    B = jnp.stack([b, 2.0 * b, -b])  # one hierarchy shared across the batch
    X = jax.vmap(lambda bb: solver(op, bb, M=M))(B)
    assert np.abs(np.asarray(X[1]) - 2.0 * u_ref).max() < 1e-8
    # exactly linear in r (fixed cycle from zero guess) -- the CG-legality requirement
    v, w = b, jnp.roll(b, 3)
    assert float(jnp.abs(M(2.0 * v + w) - 2.0 * M(v) - M(w)).max()) < 1e-12


def test_stokes_fgmres_triangular_with_amg_velocity_block():
    """The production saddle-point architecture: AMG on the (symmetric) velocity block, weighted
    pressure mass as the Schur approximation, flexible outer Krylov."""
    inner_, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    G, mu, H, Lx = 1.0, 1.0, 1.0, 4.0
    u_profile = lambda y: (G / (2 * mu)) * y * (H - y)
    d = jno.domain(box(0.0, 0.0, Lx, H), mesh_size=0.25)
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
    u_ref = np.asarray(fem.solve(linear=jno.solve.lu()))
    sol = fem.solve(
        linear=jno.solve.fgmres(tol=1e-10, restart=40, maxiter=4000),
        precond=jno.precond.triangular(
            (u, jno.precond.amg(cycles=2)),  # velocity block via the (dense) block view
            (p, jno.precond.form([(1.0 / mu) * pp * qq], inner=jno.solve.dense())),
        ),
    )
    assert np.abs(np.asarray(sol) - u_ref).max() < 1e-6


def test_traced_setup_raises_with_build_guidance():
    import jax.experimental.sparse as jsp

    from jno.utils.solver.amg import build_hierarchy

    fem = _poisson(mesh_size=0.2)
    A = fem.A if hasattr(fem.A, "todense") else jsp.BCOO.fromdense(jnp.asarray(fem.A))

    def traced(data):
        build_hierarchy(jsp.BCOO((data, A.indices), shape=A.shape))
        return data.sum()

    with pytest.raises(TypeError, match="spec.build"):
        jax.jit(traced)(A.data)


def test_pyamg_stays_optional(monkeypatch):
    """Without pyamg: jno.precond imports fine (it already did — lazy import), and using amg
    raises a clear ImportError naming the install."""
    monkeypatch.setitem(sys.modules, "pyamg", None)  # a None entry makes `import pyamg` raise
    fem = _poisson(mesh_size=0.2)
    with pytest.raises(ImportError, match="pip install pyamg"):
        jno.precond.amg().build(fem.A)


def test_built_amg_rides_the_compiled_slot_path():
    """AMG has the best asymptotics of any shipped preconditioner -- iterations are O(1) in n where
    Jacobi's grow as sqrt(n) -- and it was the one spec permanently locked out of the compiled slot
    path. Every Krylov iteration therefore dispatched a ~10-level V-cycle op by op from Python, which
    buried the entire advantage: at n=46677 a built-hierarchy AMG solve measured 625 ms against 29 ms
    for Jacobi-BiCGStab.

    The distinction that fixes it is that `traceable` depends on STATE, not on the class. Unbuilt,
    `materialize` calls pyamg on the host. Built, the levels are frozen data and `vcycle_apply` is
    pure JAX, so the applier traces. Compiled, the same comparison inverts to 3.2x in AMG's favour,
    rising to 4.6x at n=95061.
    """
    from jno.utils.solver.solver_api import _compilable

    fem = _poisson()
    A = fem.operator[0]
    cg = jno.solve.cg(tol=1e-10, maxiter=5000)

    unbuilt = jno.precond.amg()
    assert unbuilt.traceable is False, "an unbuilt hierarchy needs pyamg on the host -- must stay eager"
    assert unbuilt.key is None
    assert not _compilable(cg, unbuilt)

    built = jno.precond.amg().build(A)
    assert built.traceable is True, "a built hierarchy is frozen data and vcycle_apply is pure JAX"
    assert built.key is not None
    assert _compilable(cg, built), "a built AMG must reach the compiled path -- that is the whole point"

    # ... and compiling it must not change the answer
    u_ref = np.asarray(fem.solve(linear=jno.solve.lu()))
    u_eager = np.asarray(fem.solve(linear=cg, precond=unbuilt))
    u_compiled = np.asarray(fem.solve(linear=cg, precond=built))
    assert np.abs(u_eager - u_ref).max() < 1e-8
    assert np.abs(u_compiled - u_ref).max() < 1e-8
    assert np.abs(u_compiled - u_eager).max() < 1e-8


def test_two_amg_hierarchies_do_not_share_a_compilation():
    """The hierarchy IS the compilation, so it has to be in the cache key: two specs built from
    different operators must never share a compiled program."""
    fem_a, fem_b = _poisson(0.03), _poisson(0.05)
    a = jno.precond.amg().build(fem_a.operator[0])
    b = jno.precond.amg().build(fem_b.operator[0])
    assert a.key != b.key
    assert jno.precond.amg(cycles=1).build(fem_a.operator[0]).key != jno.precond.amg(cycles=2).build(fem_a.operator[0]).key

    # each still solves its OWN problem correctly
    for fem, spec in ((fem_a, a), (fem_b, b)):
        ref = np.asarray(fem.solve(linear=jno.solve.lu()))
        got = np.asarray(fem.solve(linear=jno.solve.cg(tol=1e-10, maxiter=5000), precond=spec))
        assert np.abs(got - ref).max() < 1e-8


def test_amg_refuses_an_indefinite_operator_instead_of_building_a_dead_hierarchy():
    """Smoothed aggregation assumes a POSITIVE diagonal -- its strength-of-connection graph and its
    smoothers both do. Handed an operator with negative diagonal entries it does not fail: it builds a
    degenerate hierarchy whose coarsest operator is NaN, or exactly zero. A zero coarse matrix has a
    perfectly good pseudo-inverse, so the V-cycle silently contributes nothing and the only symptom is
    an outer Krylov that converges slowly for no visible reason.

    Measured on the velocity block of a Navier-Stokes Newton tangent: 213 of 538 diagonal entries
    negative, coarse operator all zeros. Notably this is NOT about non-symmetry (that block is 1.9%
    non-symmetric), nor about Reynolds number, nor about the field being vector-valued -- AMG handles
    vector elasticity to 1e-11.
    """
    pytest.importorskip("pyamg")
    import numpy as _np

    n = 40
    A = _np.diag(2.0 * _np.ones(n)) - _np.diag(_np.ones(n - 1), 1) - _np.diag(_np.ones(n - 1), -1)
    A[: n // 2] *= -1.0  # flip half the rows -> half the diagonal goes negative
    spec = jno.precond.amg()
    with pytest.raises(ValueError, match="negative diagonal"):
        spec.build(A)


def _cavity_velocity_block(re, lagged, mesh_size=0.16):
    """The velocity block of a Navier-Stokes tangent, linearised two ways.

    `jno.lag` on the CONVECTING velocity is the Picard (Oseen) linearisation: the tangent then loses
    the reaction term `(du.grad)u`, which is the indefinite one.
    """
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    import numpy as _np
    from shapely.geometry import box

    inner_, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    nu = 1.0 / re
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    d.tag("lid", lambda x, y: y > 1 - 1e-9)
    d.tag("wall", lambda x, y: (y < 1e-9) | (x < 1e-9) | (x > 1 - 1e-9))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("lid", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    ub, vv = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    conv = inner_(gu, jno.lag(ub) if lagged else ub, n_contract=1)
    mom = inner_(conv, vv, n_contract=1) + nu * inner_(gu, gv, n_contract=2) - pp * trace(gv)
    fem = jno.fem(
        [
            mom,
            -qq * trace(gu),
            u(xl, yl)[0] - 16.0 * xl**2 * (1 - xl) ** 2,
            u(xl, yl)[1] - 0.0,
            u(xw, yw)[0] - 0.0,
            u(xw, yw)[1] - 0.0,
            p.pin(),
        ]
    )
    sol = _np.asarray(
        fem.solve(nonlinear=jno.solve.newton(direct=True, rtol=1e-9, atol=1e-9), linear=jno.solve.lu(backend="host"))
    )
    J = _np.asarray(fem._op.jacobian(sol).todense())
    F = J[: fem.offsets[1], : fem.offsets[1]]
    dg = _np.diag(F)
    free = _np.abs(dg - 1.0) > 1e-12  # drop the Dirichlet identity rows
    return F[_np.ix_(free, free)]


@pytest.mark.slow
def test_picard_keeps_the_momentum_block_definite_where_newton_does_not():
    """**Which linearisation you pick decides whether the momentum block has a scalable preconditioner.**

    The full Newton tangent carries the reaction term `(du.grad)u`, which is indefinite: measured, the
    smallest eigenvalue of its symmetric part goes NEGATIVE by Re = 100 (-6.4e-04), and AMG degrades
    with it (rel-resid 1.2e-03 at Re = 100, divergence by Re = 400). The Picard/Oseen block -- the
    convecting velocity lagged with `jno.lag` -- stays positive-definite (+8.5e-04 at the same Re) and
    AMG converges to 6.5e-11.

    This is why the NS preconditioning literature builds on Picard rather than Newton, and it is the
    reason `jno.precond.amg()` is usable on a momentum block at all.
    """
    import numpy as _np

    newton = _cavity_velocity_block(100.0, lagged=False)
    picard = _cavity_velocity_block(100.0, lagged=True)
    ev_n = _np.linalg.eigvalsh((newton + newton.T) / 2).min()
    ev_p = _np.linalg.eigvalsh((picard + picard.T) / 2).min()
    assert ev_n < 0, f"the Newton tangent should be indefinite at Re=100, got min eig {ev_n:.3e}"
    assert ev_p > 0, f"the Picard/Oseen block should stay definite, got min eig {ev_p:.3e}"

    # and the consequence: AMG is usable on one and not the other
    pyamg = pytest.importorskip("pyamg")
    from jno.utils.solver.amg import _to_scipy_csr

    resid = {}
    for name, F in (("newton", newton), ("picard", picard)):
        b = _np.random.default_rng(0).standard_normal(F.shape[0])
        ml = pyamg.smoothed_aggregation_solver(_to_scipy_csr(F), max_levels=10, max_coarse=50)
        x = ml.solve(b, tol=1e-10, maxiter=100)
        resid[name] = float(_np.linalg.norm(F @ x - b) / _np.linalg.norm(b))
    assert resid["picard"] < 1e-8, f"AMG must converge on the Oseen block, got {resid['picard']:.2e}"
    assert resid["picard"] < resid["newton"] / 100.0, resid
