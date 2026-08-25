"""``jno.precond.saddle()`` -- the standard saddle-point recipe as one call.

The oracle throughout is a direct factorisation of the same system: a preconditioner may change how
fast a Krylov method converges, never what it converges TO, so agreeing with the direct solve is the
only correctness statement worth making about one.
"""

import numpy as np
import pytest

import jno


def _stokes(mesh_size=0.3, mu=1.0):
    """Taylor-Hood Poiseuille channel -- the saddle system these blocks exist for."""
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    inner_, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    G, H, Lx = 1.0, 1.0, 4.0
    u_profile = lambda y: (G / (2 * mu)) * y * (H - y)  # noqa: E731
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
    return fem, u, p, mu


def _brinkman(mesh_size=0.3, mu=1.0, alpha=0.0):
    """The same channel with a Brinkman/Darcy drag ``alpha*u`` -- a reaction-dominated saddle.

    This is the regime the pressure mass alone stops standing in for: as ``alpha`` grows the Schur
    complement of ``alpha*M_u + mu*K_u`` stops looking like a mass matrix and starts looking like a
    Laplacian, which is exactly what the Cahouet-Chabard approximation interpolates between.
    """
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    inner_, grad, trace = jno.np.inner, jno.np.grad, jno.np.trace
    G, H, Lx = 1.0, 1.0, 4.0
    u_profile = lambda y: (G / (2 * mu)) * y * (H - y)  # noqa: E731
    d = jno.domain(box(0.0, 0.0, Lx, H), mesh_size=mesh_size)
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=2)
    p, q = d.fem_symbols(names=("p", "q"), order=1)
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    pp, qq = p.bind(x=xi, y=yi), q.bind(x=xi, y=yi)
    ub, vb = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    momentum = mu * inner_(gu, gv, n_contract=2) - pp * trace(gv)
    if alpha:
        momentum = momentum + alpha * inner_(ub, vb, n_contract=1)
    fem = jno.fem(
        [
            momentum,
            -qq * trace(gu),
            u(xb, yb)[0] - u_profile(yb),
            u(xb, yb)[1] - 0.0,
            p.pin(),
        ]
    )
    return fem, u, p, mu


def _poisson(mesh_size=0.4):
    """A single-field problem -- no block structure, so no saddle block to find."""
    pytest.importorskip("shapely", reason="shapely required for the box domain")
    from shapely.geometry import box

    grad, inner_ = jno.np.grad, jno.np.inner
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, v = d.fem_symbols(names=("u", "v"))
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    return jno.fem([inner_(grad(u, [xi, yi]), grad(v, [xi, yi]), n_contract=1) - 1.0 * v.bind(x=xi, y=yi), u(xb, yb) - 0.0])


# --------------------------------------------------------------------------------------
# it solves -- and agrees with a direct factorisation of the same system
# --------------------------------------------------------------------------------------
def test_saddle_matches_the_direct_oracle_on_2d_stokes():
    pytest.importorskip("pyamg", reason="pyamg required for the momentum block")
    fem, _u, _p, mu = _stokes()
    ref = fem.solve(linear=jno.solve.lu(backend="host"))
    got = fem.solve(linear=jno.solve.fgmres(tol=1e-9), precond=jno.precond.saddle(mass_weight=1.0 / mu))
    err = np.linalg.norm(np.asarray(got) - np.asarray(ref)) / np.linalg.norm(np.asarray(ref))
    assert err < 5e-3, f"saddle() drifted from the direct oracle: rel err {err:.2e}"


def test_saddle_is_the_one_line_form_of_the_explicit_triangular():
    """The shorthand must BE the explicit composition, not merely another thing that also works.

    Asserted on the composed structure rather than by comparing two ``fem.solve()`` calls: a second
    solve warm-starts from the first, so two runs of the same preconditioner legitimately stop at
    slightly different points and comparing them measures the warm start, not the preconditioner.
    """
    pytest.importorskip("pyamg", reason="pyamg required for the momentum block")
    fem, u, p, mu = _stokes()
    spec = jno.precond.saddle(mass_weight=1.0 / mu)
    composed = spec._compose(fem)
    assert type(composed).__name__ == "_Triangular"
    kinds = {idx: type(sub).__name__ for idx, sub in composed.pairs}
    assert kinds[fem.block_index(u)] == "_AMG"
    assert kinds[fem.block_index(p)] == "_Form"
    assert sorted(kinds) == list(range(len(fem.blocks)))  # every block covered exactly once


def test_a_nonflexible_krylov_is_not_the_pairing_and_fails_loudly():
    """``fgmres`` is the pairing; ``minres`` is a modelling error and ``gmres`` breaks down here.

    MINRES does not apply at all: a block-UPPER-TRIANGULAR preconditioner is nonsymmetric and
    MINRES's short recurrence assumes symmetry (measured relative residual 7.2e-01). The
    non-flexible GMRES is mathematically admissible -- the mass block is inverted exactly, so the
    preconditioner is a fixed linear operator -- but breaks down in the default float32 build, and
    measurably does so for the explicit ``triangular(...)`` composition this is shorthand for just
    as much as for ``saddle()``: that is a precision property of the pairing, not of this spec.

    Pinned because the failure is the GOOD outcome -- the residual firewall refuses rather than
    returning the plausible-but-wrong field a silent stall would give.
    """
    pytest.importorskip("pyamg", reason="pyamg required for the momentum block")
    fem, _u, _p, mu = _stokes()
    with pytest.raises(RuntimeError, match="did not solve the system"):
        fem.solve(linear=jno.solve.minres(tol=1e-9), precond=jno.precond.saddle(mass_weight=1.0 / mu))


# --------------------------------------------------------------------------------------
# the property a Schur preconditioner exists for: the iteration count stops growing with the mesh
# --------------------------------------------------------------------------------------
def _gmres_iterations(fem, spec, tol=1e-8):
    """Preconditioned GMRES iteration count for ``spec`` on ``fem``, counted directly.

    ``fem.stats['linear']`` records WHICH solver ran, not how many steps it took, so the count is
    measured here instead: assemble the system, materialise the preconditioner through the same
    context the solver would, and run scipy's GMRES with a counting callback.
    """
    import jax.numpy as jnp
    import scipy.sparse.linalg as spla

    from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

    A_j = fem.A
    A = np.asarray(A_j.todense() if hasattr(A_j, "todense") else A_j, dtype=float)
    b = np.asarray(fem.b, dtype=float).reshape(-1)
    prep = getattr(spec, "prepare", None)
    if prep is not None:
        prep(fem)
    M = materialize_precond(spec, PrecondContext(LinearOperator(A_j), fem))
    n = {"k": 0}

    def mv(v):
        # float64 explicitly: scipy probes a LinearOperator's dtype by calling it on an int8 vector,
        # which the sparse inner solve rejects outright.
        return np.array(M(jnp.asarray(np.asarray(v, dtype=float))), dtype=float, copy=True).reshape(-1)

    def cb(*_a):
        n["k"] += 1

    Mop = spla.LinearOperator(A.shape, matvec=mv, dtype=float)
    try:
        spla.gmres(A, b, M=Mop, rtol=tol, restart=200, maxiter=200, callback=cb, callback_type="pr_norm")
    except TypeError:  # scipy < 1.12 spells the tolerance `tol`
        spla.gmres(A, b, M=Mop, tol=tol, restart=200, maxiter=200, callback=cb, callback_type="pr_norm")
    return n["k"]


def test_iteration_count_is_mesh_robust():
    """Refine twice; the preconditioned iteration count must not grow the way the unpreconditioned
    one does. This is the property the pressure-mass Schur approximation exists for -- it is
    spectrally equivalent to the Schur complement, so the preconditioned spectrum does not spread as
    h falls. Asserted as a BOUND against the coarsest count, not a fixed number: the exact count
    moves with the mesh generator.
    """
    pytest.importorskip("pyamg", reason="pyamg required for the momentum block")
    counts = []
    for h in (0.5, 0.35, 0.25):
        fem, _u, _p, mu = _stokes(mesh_size=h)
        counts.append(_gmres_iterations(fem, jno.precond.saddle(mass_weight=1.0 / mu)))
    assert all(c > 0 for c in counts), f"no iterations recorded: {counts}"
    assert counts[-1] <= 2 * counts[0] + 8, f"iteration count grew with the mesh: {counts}"


# --------------------------------------------------------------------------------------
# it refuses, by name, rather than mis-solving
# --------------------------------------------------------------------------------------
def test_refuses_on_a_problem_with_no_saddle_block():
    fem = _poisson()
    with pytest.raises(ValueError, match="no saddle block"):
        fem.solve(linear=jno.solve.fgmres(tol=1e-8), precond=jno.precond.saddle())


def test_the_system_records_its_saddle_blocks_by_position():
    """`saddle()` addresses blocks by index, so the index has to be recorded, not just the name."""
    fem, _u, p, _mu = _stokes(mesh_size=0.5)
    assert fem._saddle_blocks == ("p",)
    assert fem._saddle_block_indices == (fem.block_index(p),)


def test_a_single_field_problem_records_no_saddle_block():
    fem = _poisson()
    assert fem._saddle_blocks == ()
    assert fem._saddle_block_indices == ()


def test_mass_weight_changes_speed_but_not_the_answer():
    """A wrong weight is a convergence-rate mistake, never a correctness one -- so the two solves
    must agree even though one is preconditioned badly."""
    pytest.importorskip("pyamg", reason="pyamg required for the momentum block")
    fem, _u, _p, mu = _stokes()
    ref = fem.solve(linear=jno.solve.lu(backend="host"))
    bad = fem.solve(linear=jno.solve.fgmres(tol=1e-9), precond=jno.precond.saddle(mass_weight=25.0 / mu))
    err = np.linalg.norm(np.asarray(bad) - np.asarray(ref)) / np.linalg.norm(np.asarray(ref))
    assert err < 5e-3, f"a mis-weighted saddle() changed the answer: rel err {err:.2e}"


def test_saddle_is_exported_and_reprs():
    assert "saddle" in jno.precond.__all__
    assert "mass_weight=2.0" in repr(jno.precond.saddle(mass_weight=2.0))


# --------------------------------------------------------------------------------------
# laplace_weight: the Cahouet-Chabard Schur approximation for a reaction-dominated system
# --------------------------------------------------------------------------------------
def test_laplace_weight_switches_the_schur_approximation():
    """Structural: the default builds one auxiliary, ``laplace_weight`` builds the summed pair.

    Asserted on the composition rather than by timing, for the same reason as the equivalence test
    above -- and because the thing worth pinning is that the Laplacian is *not* assembled when it was
    not asked for. An always-on Laplacian would be a silent extra factorisation on every Stokes solve.
    """
    pytest.importorskip("pyamg", reason="pyamg required for the momentum block")
    fem, u, p, mu = _brinkman(mesh_size=0.5, alpha=1e3)
    plain = jno.precond.saddle(mass_weight=1.0 / mu)._compose(fem)
    cc = jno.precond.saddle(mass_weight=1.0 / mu, laplace_weight=1e-3)._compose(fem)
    kinds = lambda c: {idx: type(sub).__name__ for idx, sub in c.pairs}  # noqa: E731
    assert kinds(plain)[fem.block_index(p)] == "_Form"
    assert kinds(cc)[fem.block_index(p)] == "_CahouetChabard"
    assert kinds(cc)[fem.block_index(u)] == "_AMG", "the momentum block must be untouched by the switch"


def test_the_pressure_laplacian_auxiliary_is_gauged():
    """The quiet failure this feature has to not have.

    A pure-Neumann Laplacian is exactly singular (the constant is a null vector), and scipy's sparse
    LU does **not** refuse it -- it factors without complaint and then applies nonsense. So the test
    is not "does it raise" but "is the operator the spec actually builds nonsingular": assemble the
    auxiliary the way ``_CahouetChabard`` does, with and without its gauge, and compare.

    Asserted as ``L @ 1`` rather than as a smallest singular value, because the null VECTOR is exact
    while the null VALUE is only zero to working precision -- and this suite runs in float32, where
    the singular value of the constant mode sits around 1e-8 and no absolute threshold separates
    "singular" from "merely ill-conditioned".
    """
    fem, _u, _p, _mu = _brinkman(mesh_size=0.5)
    d = fem.domain
    u_sym, _v_sym = d.fem_symbols()
    coords = d.variable("interior", split=True)
    axes = ("x", "y", "z")[: int(d.dimension)]
    ub = u_sym.bind(**{ax: coords[i] for i, ax in enumerate(axes)})
    vb = _v_sym.bind(**{ax: coords[i] for i, ax in enumerate(axes)})
    stiff = None
    for ax in axes:
        leg = getattr(ub, ax) * getattr(vb, ax)
        stiff = leg if stiff is None else stiff + leg

    def constant_mode_residual(terms):
        """``||L @ 1|| / (||L||_max * sqrt(n))`` -- how nearly the constant is annihilated."""
        aux = jno.fem(list(terms))
        A = np.asarray(aux.A.todense() if hasattr(aux.A, "todense") else aux.A, dtype=float)
        ones = np.ones(A.shape[1])
        return float(np.linalg.norm(A @ ones) / (np.abs(A).max() * np.sqrt(A.shape[1])))

    ungauged = constant_mode_residual([stiff])
    gauged = constant_mode_residual([stiff, u_sym.pin()])
    assert ungauged < 1e-5, f"expected the constant to be a null vector, got residual {ungauged:.2e}"
    assert gauged > 1e-3, f"the gauge did not remove the constant null vector: residual {gauged:.2e}"


def _gmres_residual_after(fem, spec, budget):
    """Relative preconditioned residual after **exactly** ``budget`` GMRES iterations.

    A fixed budget rather than a count-to-convergence, because counting is the slow direction here:
    the mass-only preconditioner needs several hundred iterations on a reaction-dominated system and
    every one of them crosses to the host through the AMG V-cycle. ``restart=budget, maxiter=1`` runs
    one cycle of exactly that many inner iterations, which bounds the work at both ends and compares
    the two approximations at equal cost.
    """
    import jax.numpy as jnp
    import scipy.sparse.linalg as spla

    from jno.utils.solver.solver_api import LinearOperator, PrecondContext, materialize_precond

    A_j = fem.A
    A = np.asarray(A_j.todense() if hasattr(A_j, "todense") else A_j, dtype=float)
    b = np.asarray(fem.b, dtype=float).reshape(-1)
    prep = getattr(spec, "prepare", None)
    if prep is not None:
        prep(fem)
    M = materialize_precond(spec, PrecondContext(LinearOperator(A_j), fem))
    hist = []

    def mv(v):
        return np.array(M(jnp.asarray(np.asarray(v, dtype=float))), dtype=float, copy=True).reshape(-1)

    Mop = spla.LinearOperator(A.shape, matvec=mv, dtype=float)
    spla.gmres(
        A,
        b,
        M=Mop,
        rtol=1e-14,
        restart=budget,
        maxiter=1,
        callback=lambda pr: hist.append(float(pr)),
        callback_type="pr_norm",
    )
    assert hist, "GMRES recorded no iterations"
    return hist[-1]


def test_the_laplacian_leg_flattens_the_brinkman_convergence():
    """The property ``laplace_weight`` exists for: convergence stops tracking the drag coefficient.

    Both preconditioners get the SAME iteration budget on the SAME system, so the only difference is
    the Schur approximation. Measured at ``alpha=1e3``, 60 iterations: the pressure mass alone is
    still at 1.1e-01 while Cahouet-Chabard has reached 1.2e-10. The thresholds below leave four
    orders of margin against that, since the exact numbers move with the mesh generator and AMG setup.

    The second assertion is the one that stops this passing for the wrong reason: without it, two
    preconditioners that BOTH converged comfortably would satisfy the ratio by accident.
    """
    pytest.importorskip("pyamg", reason="pyamg required for the momentum block")
    alpha, budget = 1.0e3, 60
    fem, _u, _p, mu = _brinkman(mesh_size=0.35, alpha=alpha)
    mass_only = _gmres_residual_after(fem, jno.precond.saddle(mass_weight=1.0 / mu), budget)
    both = _gmres_residual_after(fem, jno.precond.saddle(mass_weight=1.0 / mu, laplace_weight=1.0 / alpha), budget)
    assert mass_only > 1e-3, f"the mass-only baseline converged on its own -- no contrast to measure ({mass_only:.2e})"
    assert both < 1e-4 * mass_only, (
        f"the Laplacian leg did not help at alpha={alpha:g}: {mass_only:.2e} -> {both:.2e} in {budget} iterations"
    )


def test_cahouet_chabard_matches_the_direct_oracle():
    """A richer Schur approximation is still only a preconditioner -- same answer, sooner."""
    pytest.importorskip("pyamg", reason="pyamg required for the momentum block")
    alpha = 1.0e3
    fem, _u, _p, mu = _brinkman(alpha=alpha)
    ref = fem.solve(linear=jno.solve.lu(backend="host"))
    got = fem.solve(
        linear=jno.solve.fgmres(tol=1e-9, restart=150),
        precond=jno.precond.saddle(mass_weight=1.0 / mu, laplace_weight=1.0 / alpha),
    )
    err = np.linalg.norm(np.asarray(got) - np.asarray(ref)) / np.linalg.norm(np.asarray(ref))
    assert err < 5e-3, f"Cahouet-Chabard drifted from the direct oracle: rel err {err:.2e}"


def test_laplace_weight_reprs_and_is_absent_by_default():
    assert "laplace_weight" not in repr(jno.precond.saddle(mass_weight=2.0))
    assert "laplace_weight=0.001" in repr(jno.precond.saddle(mass_weight=2.0, laplace_weight=1e-3))
