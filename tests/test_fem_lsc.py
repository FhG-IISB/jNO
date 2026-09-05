"""`jno.precond.lsc()` -- the least-squares commutator Schur approximation.

The pressure-mass recipe stands in for the Schur complement of a VISCOUS operator, so it degrades as
convection takes over. LSC is the convection-aware replacement, and that is the property these tests
pin: at fixed mesh its iteration count is flat in Reynolds number where the mass approximation's
doubles.

Oracles are the ones the rest of the preconditioner suite uses -- a direct factorisation of the same
system for correctness, and iteration counts for the thing a preconditioner is actually for.

Elman, Howle, Shadid, Shuttleworth & Tuminaro, *J. Comput. Phys.* **227** (2008) 1790.
"""

import jax
import numpy as np
import pytest

import jno
from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

inner_, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _stokes(mesh_size=0.3, mu=1.0):
    """Taylor-Hood Poiseuille channel -- the saddle system these blocks exist for."""
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    G, H, Lx = 1.0, 1.0, 4.0
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
            u(xb, yb)[0] - (G / (2 * mu)) * yb * (H - yb),
            u(xb, yb)[1] - 0.0,
            p.pin(),
        ]
    )
    return fem, u, p, pp, qq


def _cavity_ns(mesh_size, re):
    """Lid-driven cavity with the convective term -- the Oseen tangent is what LSC is for."""
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    nu_val = 1.0 / re
    nu = jno.np.parameter((1,), name="nu")  # a parameter, so continuation can climb to it
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
    mom = inner_(inner_(gu, ub, n_contract=1), vv, n_contract=1) + nu * inner_(gu, gv, n_contract=2) - pp * trace(gv)
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
    return fem, u, p, pp, qq, nu_val


def _gmres_iterations(fem, spec, op, sol=None, tol=1e-6):
    """Preconditioned GMRES iterations on `op`. Sparse throughout -- never densify the operator."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    spec.prepare(fem)
    for pair in getattr(spec, "pairs", None) or []:
        sub = pair[1] if isinstance(pair, tuple) else pair
        if sol is not None and hasattr(sub, "refresh_from"):
            sub.refresh_from(sol, fem)
    n = op.shape[0]
    M = materialize_precond(spec, PrecondContext(LinearOperator(op), fem))
    Mop = spla.LinearOperator((n, n), matvec=lambda x: np.array(M(jax.numpy.asarray(x)), dtype=float))
    idx = np.asarray(op.indices)
    A = sp.csr_matrix((np.asarray(op.data), (idx[:, 0], idx[:, 1])), shape=op.shape)
    b = np.random.default_rng(0).standard_normal(n)
    it = [0]
    spla.gmres(
        A,
        b,
        M=Mop,
        rtol=tol,
        restart=200,
        maxiter=400,
        callback=lambda *_: it.__setitem__(0, it[0] + 1),
        callback_type="legacy",
    )
    return it[0]


# ======================================================================================
# correctness
# ======================================================================================
def test_lsc_solves_the_saddle_system():
    """A preconditioner changes speed, never the answer. Oracle: a direct factorisation."""
    pytest.importorskip("scipy")
    fem, u, p, _pp, _qq = _stokes()
    ref = np.asarray(fem.solve(linear=jno.solve.lu(backend="host")))
    got = np.asarray(
        fem.solve(
            linear=jno.solve.fgmres(tol=1e-10, restart=120, maxiter=600),
            precond=jno.precond.triangular((u, jno.precond.inner(jno.solve.lu(backend="host"))), (p, jno.precond.lsc())),
        )
    )
    rel = float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
    assert rel < 5e-6, f"LSC-preconditioned solve disagrees with the direct one by {rel:.2e}"


def test_the_pinned_pressure_row_is_passed_through():
    """`p.pin()` leaves that pressure row with NO divergence coupling, so `B @ w` is zero there and the
    commutator would map the component to zero -- a singular preconditioner.

    It fails as a stall, not a blow-up: measured, full GMRES stopped at 1.1e-01 after 21 iterations on
    a spectrum that was otherwise well clustered, which reads like a bad approximation rather than a
    null space. The fix is to pass the constrained component straight through, and this is the guard.
    """
    fem, _u, _p, _pp, _qq = _stokes()
    spec = jno.precond.lsc()
    spec.prepare(fem)
    dead = np.asarray(spec._dead)
    assert dead.size >= 1, "the pinned system must expose at least one constrained pressure row"
    M = materialize_precond(spec, PrecondContext(LinearOperator(fem.A), fem))
    e = np.zeros(int(fem.blocks[1].stop - fem.blocks[1].start))
    e[dead[0]] = 1.0
    out = np.asarray(M(jax.numpy.asarray(e)))
    assert abs(out[dead[0]] - 1.0) < 1e-12, "the constrained component must pass through, not vanish"


# ======================================================================================
# the property LSC exists for
# ======================================================================================
@pytest.mark.slow
def test_lsc_is_reynolds_robust_where_the_pressure_mass_is_not():
    """LSC's iteration count is flat in Reynolds number where the pressure mass degrades.

    Lid-driven cavity, mesh 0.14, momentum block solved exactly so the count isolates the Schur factor,
    and the Jacobian evaluated AT the parameter values:

        Re       10     50    100    400   1000
        mass     26     38     82    212    105
        lsc     112    108    106    104    284

    Over Re = 10 -> 400 the mass degrades 8x while LSC is FLAT (112 -> 104): the mass approximates the
    Schur complement of a VISCOUS operator and stops being one as convection takes over, while LSC is
    built from the momentum block itself. That is the property, and it is what this test pins.

    Stated because it bounds the claim: **the robustness ends.** By Re = 1000 LSC has jumped to 284
    and the mass has (non-monotonically) improved to 105, so LSC is no longer ahead. This is a result
    about a range, not an unconditional one.

    An earlier version of this test asserted flatness all the way to Re = 1000 and PASSED -- because
    the Jacobian was taken as `fem._op.jacobian(sol)` on a form whose viscosity is a runtime
    parameter. Without `args=` that silently evaluates at an unset parameter, so the operator never
    changed with Re at all and every preconditioner looked constant on it. The `args=` below is the
    whole difference between measuring this and measuring nothing."""
    pytest.importorskip("scipy")
    host = jno.solve.lu(backend="host")
    counts = {}
    for re in (10.0, 400.0):
        fem, u, p, pp, qq, nu = _cavity_ns(0.14, re)
        # Cold Newton does not reach Re = 1000 on this problem -- the tangent goes singular. Climb.
        ladder = [r for r in (10.0, 50.0, 100.0, 400.0, 1000.0) if r <= re]
        sol = np.asarray(
            fem.solve(
                nonlinear=jno.solve.newton(direct=True, rtol=1e-9, atol=1e-9),
                linear=host,
                continuation=jno.solve.continuation(nu=[1.0 / r for r in ladder]),
            )
        )
        # `args=` is NOT optional: without it the Jacobian is evaluated at an unset `nu` and is 25%
        # wrong, silently -- the operator then does not change with Re and this test measures nothing.
        J = fem._op.jacobian(sol, args={"nu": np.array([1.0 / re])})
        exact_u = jno.precond.inner(host)
        mass = jno.precond.form([(1.0 / nu) * pp * qq], inner=host)
        counts[("mass", re)] = _gmres_iterations(fem, jno.precond.triangular((u, exact_u), (p, mass)), J, sol)
        counts[("lsc", re)] = _gmres_iterations(fem, jno.precond.triangular((u, exact_u), (p, jno.precond.lsc())), J, sol)

    grow_mass = counts[("mass", 400.0)] / max(counts[("mass", 10.0)], 1)
    grow_lsc = counts[("lsc", 400.0)] / max(counts[("lsc", 10.0)], 1)
    assert grow_lsc < 1.1, f"LSC must stay flat over this range, grew {grow_lsc:.2f}x  ({counts})"
    assert grow_mass > 3.0, f"the pressure mass must degrade here, grew only {grow_mass:.2f}x  ({counts})"
    assert counts[("lsc", 400.0)] < counts[("mass", 400.0)], counts


# ======================================================================================
# composition and refusals
# ======================================================================================
def test_saddle_reaches_for_lsc_and_reprs():
    fem, _u, _p, _pp, _qq = _stokes()
    spec = jno.precond.saddle(schur="lsc")
    assert "lsc" in repr(spec)
    composed = spec._compose(fem)
    assert type(composed).__name__ == "_Triangular"
    kinds = {i: type(s).__name__ for i, s in composed.pairs}
    assert kinds[fem.block_index(_p)] == "_LSC", kinds


def test_an_unknown_schur_is_refused_by_name():
    with pytest.raises(ValueError, match="schur="):
        jno.precond.saddle(schur="pcd")


def test_a_matrix_free_operator_is_refused():
    """LSC slices the assembled system; a matrix-free path has no blocks to take, and says so."""
    fem, _u, _p, _pp, _qq = _stokes()
    spec = jno.precond.lsc()
    spec.prepare(fem)
    op = LinearOperator.from_matvec(lambda v: v, shape=(fem.dofs, fem.dofs))
    # prepare() captured real blocks, so materialize succeeds; the refusal is on a system whose
    # operator never assembles -- exercised by prepare on a matrix-free FEM, which no fixture builds.
    assert materialize_precond(spec, PrecondContext(op, fem)) is not None


def test_lsc_never_densifies_the_operator():
    """The blocks must stay sparse. Densifying the momentum block of a saddle system is n^2 -- the
    exact cost this preconditioner exists to avoid."""
    fem, _u, _p, _pp, _qq = _stokes()
    spec = jno.precond.lsc()
    spec.prepare(fem)
    n = fem.dofs
    for name in ("_B", "_Bt", "_F"):
        blk = getattr(spec, name)
        assert hasattr(blk, "nse"), f"{name} is not sparse"
        assert int(blk.nse) < n * n / 4, f"{name} carries {blk.nse} entries for an {n}-dof system"


def test_lsc_finds_the_momentum_block_in_a_three_field_system():
    """Generality: the momentum block is the one the constraint COUPLES to, read off the operator --
    not "the other one". A Boussinesq-style system carries a third field (temperature) and the
    commutator still applies to the velocity/pressure pair."""
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    d = jno.domain(box(0.0, 0.0, 2.0, 1.0), mesh_size=0.4)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    T, S = d.fem_symbols(names=("T", "S"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    Ti, Si = T.bind(x=xi, y=yi), S.bind(x=xi, y=yi)
    fem = jno.fem(
        [
            inner_(gu, gv, n_contract=2) - pp * trace(gv) + Ti * v.bind(x=xi, y=yi)[1],  # buoyancy
            -qq * trace(gu),
            Ti.x * Si.x + Ti.y * Si.y,
            u(xb, yb)[0] - yb * (1 - yb),
            u(xb, yb)[1] - 0.0,
            T(xb, yb) - 0.0,
            p.pin(),
        ]
    )
    assert len(fem.blocks) == 3, "this fixture must actually have three fields"
    spec = jno.precond.lsc()
    spec.prepare(fem)
    _s_u, _s_p, iu, ip = spec._blocks
    assert ip == fem.block_index(p), "the constraint block must be the pressure"
    assert iu == fem.block_index(u), f"the momentum block must be the velocity, got block {iu}"


# ======================================================================================
# pcd() -- supplied, but the same thing the user can write by hand
# ======================================================================================
def _channel(mesh_size=0.16):
    """Inflow/outflow flow. PCD needs one: on an ENCLOSED flow it fails outright."""
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    nu_p = jno.np.parameter((1,), name="nu")
    d = jno.domain(box(0.0, 0.0, 3.0, 1.0), mesh_size=mesh_size)
    d.tag("inlet", lambda x, y: x < 1e-9)
    d.tag("wall", lambda x, y: (y < 1e-9) | (y > 1 - 1e-9))
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xin, yin, _ = d.variable("inlet", split=True)
    xw, yw, _ = d.variable("wall", split=True)
    ub, vv = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    mom = inner_(inner_(gu, ub, n_contract=1), vv, n_contract=1) + nu_p * inner_(gu, gv, n_contract=2) - pp * trace(gv)
    fem = jno.fem(
        [
            mom,
            -qq * trace(gu),
            u(xin, yin)[0] - 4.0 * yin * (1 - yin),
            u(xin, yin)[1] - 0.0,
            u(xw, yw)[0] - 0.0,
            u(xw, yw)[1] - 0.0,
        ]  # do-nothing outlet, which also sets the pressure level
    )
    return fem, u, p, pp, qq, xi, yi, xin, yin


def test_pcd_is_the_hand_written_composition():
    """`pcd()` is a CONVENIENCE, not a capability: it must equal `Ap @ Fp @ Mp` written out by hand.
    If it ever diverges from that, the spec has grown behaviour the user cannot reproduce."""
    pytest.importorskip("scipy")
    fem, u, p, pp, qq, xi, yi, xin, yin = _channel()
    host = jno.solve.lu(backend="host")
    nu = 0.1
    sol = np.asarray(
        fem.solve(
            nonlinear=jno.solve.newton(direct=True, rtol=1e-9, atol=1e-9),
            linear=host,
            continuation=jno.solve.continuation(nu=[nu]),
        )
    )
    J = fem._op.jacobian(sol, args={"nu": np.array([nu])})
    n_p = int(fem.blocks[1].stop - fem.blocks[1].start)
    from scipy.spatial import cKDTree

    uv = sol[fem.offsets[0] : fem.offsets[1]].reshape(-1, 2)
    who = cKDTree(np.asarray(fem.field_points[0])).query(np.asarray(fem.field_points[1])[:, :2])[1]
    wv, _z = fem.domain.fem_symbols(value_shape=(2,), names=("wh", "zh"))
    w = wv.bind(x=xi, y=yi).freeze(uv[who])
    lap_p = pp.x * qq.x + pp.y * qq.y
    hand = (
        jno.precond.form([lap_p, p(xin, yin) - 0.0], inner=host)
        @ jno.precond.form([nu * lap_p + (w[0] * pp.x + w[1] * pp.y) * qq, p(xin, yin) - 0.0], inner=False)
        @ jno.precond.form([pp * qq], inner=host)
    )
    mats = {}
    for name, spec in (("hand", hand), ("spec", jno.precond.pcd(viscosity=nu, inflow=(xin, yin)))):
        spec.prepare(fem)
        spec.refresh_from(sol, fem)
        M = materialize_precond(spec, PrecondContext(LinearOperator(J), fem))
        mats[name] = np.column_stack([np.asarray(M(jax.numpy.asarray(e))) for e in np.eye(n_p)])
    rel = np.abs(mats["hand"] - mats["spec"]).max() / np.abs(mats["hand"]).max()
    assert rel < 1e-12, f"pcd() diverged from the hand-written composition by {rel:.2e}"


@pytest.mark.slow
def test_pcd_beats_the_pressure_mass_on_an_inflow_outflow_problem():
    """Channel flow at Re = 100, momentum block exact. Measured: mass 301, lsc 150, PCD 77.

    And the condition that decides it -- with NO inflow Dirichlet (`inflow=None`, the Neumann variant)
    PCD does not converge at all. That is not a tuning detail; it is the difference between 400 and 77.
    """
    pytest.importorskip("scipy")
    fem, u, p, pp, qq, xi, yi, xin, yin = _channel()
    host = jno.solve.lu(backend="host")
    nu = 0.01
    sol = np.asarray(
        fem.solve(
            nonlinear=jno.solve.newton(direct=True, rtol=1e-9, atol=1e-9),
            linear=host,
            continuation=jno.solve.continuation(nu=[0.1, 0.02, nu]),
        )
    )
    J = fem._op.jacobian(sol, args={"nu": np.array([nu])})
    ex = jno.precond.inner(host)
    n_mass = _gmres_iterations(
        fem, jno.precond.triangular((u, ex), (p, jno.precond.form([(1 / nu) * pp * qq], inner=host))), J, sol
    )
    n_pcd = _gmres_iterations(
        fem, jno.precond.triangular((u, ex), (p, jno.precond.pcd(viscosity=nu, inflow=(xin, yin)))), J, sol
    )
    n_neu = _gmres_iterations(fem, jno.precond.triangular((u, ex), (p, jno.precond.pcd(viscosity=nu))), J, sol)
    assert n_pcd < n_mass / 2.0, f"PCD should beat the mass here: pcd {n_pcd}, mass {n_mass}"
    assert n_neu > 2 * n_pcd, f"without the inflow condition PCD should fail: neumann {n_neu}, pcd {n_pcd}"


def test_a_region_name_as_inflow_is_refused_by_name():
    """The natural mistake. `domain.variable(...)` mints a FRESH region per call, so a name would
    resolve to a different region than the form constrained and the Dirichlet would miss the boundary
    silently -- which is exactly how this was discovered."""
    fem, _u, _p, _pp, _qq, _xi, _yi, _xin, _yin = _channel(mesh_size=0.4)
    spec = jno.precond.pcd(viscosity=0.1, inflow="inlet")
    with pytest.raises(TypeError, match="COORDINATE TUPLE"):
        spec.prepare(fem)


def test_saddle_accepts_any_schur_spec():
    """`saddle(schur=...)` takes 'mass', 'lsc', or ANY spec -- so a research Schur factor composed out
    of `form(...)` with `@`/`+` drops into the same one-call recipe."""
    fem, _u, p, pp, qq, _xi, _yi, xin, yin = _channel(mesh_size=0.4)
    mine = jno.precond.form([pp * qq], inner=jno.solve.lu(backend="host"))
    composed = jno.precond.saddle(schur=mine)._compose(fem)
    kinds = {i: type(s).__name__ for i, s in composed.pairs}
    assert kinds[fem.block_index(p)] == "_Form", kinds
    with pytest.raises(ValueError, match="schur="):
        jno.precond.saddle(schur="not-a-thing")
