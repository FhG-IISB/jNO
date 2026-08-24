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
