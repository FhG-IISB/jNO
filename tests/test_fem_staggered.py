"""``jno.solve.staggered([u, d])`` — alternate minimization as a nonlinear slot.

Solve a coupled system one field at a time, sweeping until the FULL residual converges, instead of
driving all fields together in one Newton. It exists for energies that are **non-convex in the fields
jointly but convex in each separately**, where a monolithic Newton has no descent guarantee: the
canonical case is variational phase-field fracture, whose ``(1-d)^2 |grad u|^2`` coupling is quartic in
the pair while each field's own problem is linear elliptic.

Oracles:
* **agreement on a convex problem** — staggered and monolithic must land on the same solution to solver
  tolerance when both converge. This is the correctness oracle: alternate minimization is a different
  *route* to the same root, not a different answer.
* **it converges where monolithic does not** — the motivating case, measured: the same non-convex
  phase-field energy that makes ``newton()`` diverge (residual ~1e25) is solved by the staggered sweep,
  and the answer satisfies the coupled residual.
* **differentiability** — the gradient w.r.t. a material parameter flows through the sweep and matches
  finite differences. The alternating structure is absent from the derivative by construction (the
  implicit-function theorem acts at the root), which is exactly what this checks.
* **fail-loud** — an unlisted field, a duplicate, a single-field problem, and use off ``fem.solve``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jno


def _aliases():
    n = jno.np
    return n.grad, n.inner


def _convex_pair(size=0.12):
    """Two Poisson fields coupled BOTH ways, weakly enough to stay jointly convex:

        -Δa + a = 1 + k·b        -Δb + b = 2·a

    Linear and coercive, so it has one solution and both drivers must find it."""
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1]]
    a, phi = d.fem_symbols()
    b, chi = d.fem_symbols()
    terms = [
        inner(grad(a, X), grad(phi, X), 1) + a * phi - 1.0 * phi - 0.3 * b * phi,
        inner(grad(b, X), grad(chi, X), 1) + b * chi - 2.0 * a * chi,
        # a nonlinearity so the problem takes the residual (Newton) route at all
        0.05 * a * a * phi,
        a(*cb) - 0.0,
        b(*cb) - 0.0,
    ]
    return jno.fem(terms), a, b


# --------------------------------------------------------------------------------------------------
# Oracle 1 — correctness: on a problem where BOTH converge, they must agree. A different route to the
# same root, not a different answer.
# --------------------------------------------------------------------------------------------------
def test_staggered_and_monolithic_agree_on_a_convex_problem():
    fem, a, b = _convex_pair()
    mono = np.asarray(fem.solve(nonlinear=jno.solve.newton(rtol=1e-11, atol=1e-13)))
    stag = np.asarray(fem.solve(nonlinear=jno.solve.staggered([a, b], rtol=1e-11, atol=1e-13)))
    assert np.abs(mono).max() > 1e-3, "the solution is trivial — the comparison would be vacuous"
    rel = np.abs(stag - mono).max() / np.abs(mono).max()
    assert rel < 1e-8, f"staggered and monolithic disagree: rel {rel:.3e}"
    # ...and both blocks are genuinely nonzero, so the agreement is not one block matching zeros.
    for blk in fem.blocks:
        assert np.abs(mono[blk]).max() > 1e-3


def test_sweep_order_does_not_change_the_solution():
    """Gauss-Seidel order changes the ITERATION, not the root."""
    fem, a, b = _convex_pair()
    ab = np.asarray(fem.solve(nonlinear=jno.solve.staggered([a, b], rtol=1e-11, atol=1e-13)))
    ba = np.asarray(fem.solve(nonlinear=jno.solve.staggered([b, a], rtol=1e-11, atol=1e-13)))
    assert np.abs(ab - ba).max() / np.abs(ab).max() < 1e-8


# --------------------------------------------------------------------------------------------------
# Oracle 2 — the motivating case: a non-convex phase-field energy that a monolithic Newton cannot
# solve. `(1-d)^2 |grad u|^2` is quartic in the pair; each field alone is linear elliptic. Measured on
# this exact problem: newton() leaves with a residual around 1e25 (it diverges), while the staggered
# sweep converges — and the answer is checked against the COUPLED residual, not against itself.
# --------------------------------------------------------------------------------------------------
def _phase_field(load=2.0, ell=0.4, gc=1.0, eta=1e-4, size=0.34):
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=size).domain()
    d.tag("left", lambda x, y: x < 1e-9)
    co, cl = d.variable("interior", split=True), d.variable("left", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    dm, q = d.fem_symbols()
    psi = 0.5 * inner(grad(u, X), grad(u, X), 1)
    momentum = ((1.0 - dm) ** 2 + eta) * inner(grad(u, X), grad(phi, X), 1) - load * phi
    damage = (gc / ell) * dm * q + gc * ell * inner(grad(dm, X), grad(q, X), 1) - 2.0 * (1.0 - dm) * psi * q
    return jno.fem([momentum, damage, u(*cl) - 0.0]), u, dm, momentum, damage


def test_staggered_solves_the_non_convex_energy_that_monolithic_cannot():
    fem, u, dm, momentum, damage = _phase_field()

    # The monolithic driver fails on this energy — loudly, which is the point of the fail-loud rule.
    with pytest.raises(RuntimeError, match="did not converge"):
        fem.solve(nonlinear=jno.solve.newton(max_steps=200, line_search=True))

    # Tolerance set to what the arithmetic actually supports here: the degraded operator is nearly
    # singular where the damage saturates, and alternate minimization converges linearly, so pushing
    # for ~1e-10 stalls (measured: 1.25e-9 on GPU, reached on CPU — a tolerance, not a correctness,
    # difference). The residual checks below are the real oracle and are unaffected.
    sol = np.asarray(fem.solve(nonlinear=jno.solve.staggered([u, dm], max_sweeps=600, rtol=1e-7, atol=1e-9)))
    dmg = sol[fem.blocks[fem.block_index(dm)]]
    disp = sol[fem.blocks[fem.block_index(u)]]
    assert np.abs(disp).max() > 1e-3, "the displacement never responded"
    assert dmg.max() > 1e-2, "no damage developed — the coupling did nothing"

    # The oracle is the COUPLED residual: a staggered sweep is only correct if its fixed point is a root
    # of the whole system, not just of each block in turn.
    Ru = np.asarray(fem.eval(momentum, sol))
    Rd = np.asarray(fem.eval(damage, sol))
    x = np.asarray(fem.field_points[fem.block_index(u)])[:, 0]
    free_u = np.zeros_like(Ru, dtype=bool)
    clamped = np.zeros_like(Ru, dtype=bool)
    free_u[fem.blocks[fem.block_index(u)]] = x > 1e-9  # the clamped edge carries its reaction instead
    clamped[fem.blocks[fem.block_index(u)]] = x <= 1e-9
    assert np.abs(Ru[free_u]).max() < 1e-7, f"momentum is not satisfied: {np.abs(Ru[free_u]).max():.3e}"
    assert np.abs(Rd).max() < 1e-7, f"the damage equation is not satisfied: {np.abs(Rd).max():.3e}"
    # ...and the reaction at the clamped edge is NOT small, so neither check passes by everything
    # being zero. (It must also balance the applied load — the body is in equilibrium.)
    assert np.abs(Ru[clamped]).max() > 1e-3, "the clamped edge carries no reaction"
    assert abs(Ru[clamped].sum() + 2.0 * 1.0) < 1e-6, "the reaction does not balance the applied load"


# --------------------------------------------------------------------------------------------------
# Oracle 3 — differentiability. The alternating structure must be absent from the derivative: at the
# root the implicit-function theorem acts on the FULL Jacobian regardless of how the root was reached.
# --------------------------------------------------------------------------------------------------
def test_gradient_flows_through_the_sweep():
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1]]
    kP = jno.np.reshape(jno.np.parameter((1,), name="k"), ())
    a, phi = d.fem_symbols()
    b, chi = d.fem_symbols()
    fem = jno.fem(
        [
            inner(grad(a, X), grad(phi, X), 1) + a * phi - 1.0 * phi - kP * b * phi + 0.05 * a * a * phi,
            inner(grad(b, X), grad(chi, X), 1) + b * chi - 2.0 * a * chi,
            a(*cb) - 0.0,
            b(*cb) - 0.0,
        ]
    )
    op = fem._op
    assert op.is_parametric
    spec = jno.solve.staggered([a, b], rtol=1e-12, atol=1e-14)
    spec.prepare(fem)  # what fem.solve does at compose time
    from jno.utils.solver.newton_krylov import staggered_newton

    blocks = [np.arange(int(s.start), int(s.stop)) for s in fem.blocks]
    u0 = jnp.zeros(int(op.size))

    def norm_at(kv):
        args = {"k": jnp.reshape(kv, (1,))}
        sol = staggered_newton(lambda z: op.residual(z, args), u0, blocks, rtol=1e-12, atol=1e-14)
        return jnp.sum(sol**2)

    g = jax.grad(norm_at)(0.3)
    fd = (norm_at(0.3 + 1e-6) - norm_at(0.3 - 1e-6)) / 2e-6
    assert np.isfinite(g) and abs(g) > 0, "the gradient vanished through the sweep"
    assert np.allclose(g, fd, rtol=1e-6), f"AD {g:.8e} vs FD {fd:.8e}"


# --------------------------------------------------------------------------------------------------
# Oracle 4 — fail loud.
# --------------------------------------------------------------------------------------------------
def test_an_unlisted_field_fails_loud():
    fem, a, _b = _convex_pair(size=0.3)
    with pytest.raises(ValueError, match="every field block|not listed|never be solved"):
        fem.solve(nonlinear=jno.solve.staggered([a]))


def test_a_repeated_field_fails_loud():
    fem, a, _b = _convex_pair(size=0.3)
    with pytest.raises(ValueError, match="twice"):
        fem.solve(nonlinear=jno.solve.staggered([a, a]))


def test_a_single_field_problem_fails_loud():
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) + 0.1 * u * u * phi - 1.0 * phi, u(*cb) - 0.0])
    with pytest.raises(ValueError, match="single field block|nothing to alternate"):
        fem.solve(nonlinear=jno.solve.staggered([u]))


def test_unprepared_spec_fails_loud():
    """Used as a bare callable there is no block layout, so it must refuse rather than guess."""
    spec = jno.solve.staggered([0, 1])
    with pytest.raises(ValueError, match="block layout|fem.solve"):
        spec(lambda z: z, jnp.zeros(4))


# --------------------------------------------------------------------------------------------------
# Oracle 5 — `direct=True`: factorize each field's ASSEMBLED diagonal block instead of solving it
# matrix-free. It exists because the matrix-free sub-solve cannot be preconditioned — a `precond=` spec
# materializes against an assembled operator and a restriction closure has none — so an ill-conditioned
# block (near-incompressible elasticity) is solved by UNPRECONDITIONED BiCGStab.
#
# The correctness oracle is the same as everywhere else here: a different route to the same root.
# --------------------------------------------------------------------------------------------------
def test_direct_and_matrix_free_staggered_agree():
    fem, a, b = _convex_pair()
    mf = np.asarray(fem.solve(nonlinear=jno.solve.staggered([a, b], rtol=1e-11, atol=1e-13)))
    dr = np.asarray(fem.solve(nonlinear=jno.solve.staggered([a, b], rtol=1e-11, atol=1e-13, direct=True)))
    assert np.abs(mf).max() > 1e-3, "the solution is trivial — the comparison would be vacuous"
    rel = np.abs(dr - mf).max() / np.abs(mf).max()
    assert rel < 1e-8, f"direct and matrix-free staggered disagree: rel {rel:.3e}"
    for blk in fem.blocks:
        assert np.abs(dr[blk]).max() > 1e-3, "a block came back zero — the sub-block solve did nothing"


def test_direct_staggered_solves_the_non_convex_energy_too():
    """The motivating problem still converges on the direct route — `direct=` changes the LINEAR
    algebra inside each sub-solve, not the alternating structure that makes it converge at all."""
    fem, u, dm, momentum, damage = _phase_field()
    # Same settings the matrix-free test above establishes for this problem: the degraded operator is
    # nearly singular where the damage saturates and alternate minimization converges linearly, so ~1e-10
    # stalls. That is a property of the OUTER sweep, which `direct=` does not change.
    kw = dict(max_sweeps=600, rtol=1e-7, atol=1e-9)
    mf = np.asarray(fem.solve(nonlinear=jno.solve.staggered([u, dm], **kw)))
    dr = np.asarray(fem.solve(nonlinear=jno.solve.staggered([u, dm], direct=True, **kw)))
    assert np.abs(mf).max() > 1e-3
    assert np.asarray(mf)[fem.blocks[fem.block_index(dm)]].max() > 1e-2, "no damage — the coupling did nothing"
    assert np.abs(dr - mf).max() / np.abs(mf).max() < 1e-6
    # The oracle is the COUPLED residual, as above: a sweep is correct only if its fixed point is a root
    # of the whole system.
    assert np.abs(np.asarray(fem.eval(damage, dr))).max() < 1e-7


def test_direct_staggered_composes_with_a_direct_linear_slot():
    """`linear=lu()` used to be REFUSED against staggered (the composer rejects a direct linear slot
    paired with a matrix-free nonlinear one). With `direct=True` there is an assembled block to give it."""
    fem, a, b = _convex_pair()
    ref = np.asarray(fem.solve(nonlinear=jno.solve.staggered([a, b], rtol=1e-11, atol=1e-13)))
    got = np.asarray(
        fem.solve(
            nonlinear=jno.solve.staggered([a, b], rtol=1e-11, atol=1e-13, direct=True),
            linear=jno.solve.lu(),
        )
    )
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-8


def test_matrix_free_staggered_still_refuses_a_direct_linear_slot():
    """...and without `direct=True` the refusal must stand — there is no matrix to factorize."""
    fem, a, b = _convex_pair(size=0.3)
    with pytest.raises(ValueError, match="DIRECT solver and needs the assembled tangent"):
        fem.solve(nonlinear=jno.solve.staggered([a, b]), linear=jno.solve.lu())


def test_direct_staggered_without_an_assembled_tangent_fails_loud():
    """Driven with no tangent supplied it must NAME the gap, not silently fall back to matrix-free —
    which would quietly undo the whole reason the caller asked for `direct=True`."""
    fem, a, b = _convex_pair(size=0.3)
    spec = jno.solve.staggered([a, b], direct=True)
    spec.prepare(fem)  # resolve the blocks, so the failure is the MISSING JACOBIAN and nothing else
    op = fem._op
    with pytest.raises(ValueError, match="ASSEMBLED diagonal block"):
        spec(lambda z: op.residual(z, None), jnp.zeros(int(op.size)))  # no jacobian= supplied


def test_direct_staggered_gradient_matches_finite_differences():
    """Differentiability is a requirement, not a nice-to-have. The direct route hangs `custom_root` off
    the root it found and solves the tangent (and its TRANSPOSE, for reverse mode) directly."""
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.25).domain()
    d.tag("bdry", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cb = d.variable("interior", split=True), d.variable("bdry", split=True)
    X = [co[0], co[1]]
    kP = jno.np.reshape(jno.np.parameter((1,), name="k"), ())
    a, phi = d.fem_symbols()
    b, chi = d.fem_symbols()
    fem = jno.fem(
        [
            inner(grad(a, X), grad(phi, X), 1) + a * phi - 1.0 * phi - kP * b * phi + 0.05 * a * a * phi,
            inner(grad(b, X), grad(chi, X), 1) + b * chi - 2.0 * a * chi,
            a(*cb) - 0.0,
            b(*cb) - 0.0,
        ]
    )
    op = fem._op
    from jno.utils.solver.newton_krylov import staggered_newton

    blocks = [np.arange(int(s.start), int(s.stop)) for s in fem.blocks]
    u0 = jnp.zeros(int(op.size))

    def norm_at(kv):
        args = {"k": jnp.reshape(kv, (1,))}
        sol = staggered_newton(
            lambda z: op.residual(z, args),
            u0,
            blocks,
            rtol=1e-12,
            atol=1e-14,
            jacobian=lambda z: op.jacobian(z, args),
        )
        return jnp.sum(sol**2)

    g = jax.grad(norm_at)(0.3)
    fd = (norm_at(0.3 + 1e-6) - norm_at(0.3 - 1e-6)) / 2e-6
    assert np.isfinite(g) and abs(g) > 0, "the gradient vanished through the direct sweep"
    assert np.allclose(g, fd, rtol=1e-6), f"AD {g:.8e} vs FD {fd:.8e}"


# --------------------------------------------------------------------------------------------------
# Oracle 6 — `over_relax` (ORAM). Alternate minimization IS a nonlinear block Gauss-Seidel iteration,
# so over-relaxation accelerates it the same way it accelerates the linear one: go `omega` times as far
# along each sub-step's own update direction. Farrell & Maurini, IJNME 109 (2017) 648-667, Algorithm 2,
# section 2.1.
#
# It changes the ITERATION, not the root — which is what these pin. Whether it pays is problem
# dependent and the paper says so outright (their Table I: 58-73% fewer iterations on a propagating
# crack; their Table II: it *hinders* convergence where AM was already fast). So there is no default
# worth guessing, and omega=1 must stay exactly what it was.
# --------------------------------------------------------------------------------------------------
def test_over_relax_one_is_the_unrelaxed_iteration():
    """omega = 1 must be the historic path: same iteration, and no projector requested.

    Not asserted bitwise — measured, two IDENTICAL solves differ by ~1e-17 on GPU (non-deterministic
    reduction order), so bit-equality would be testing the backend, not this change. Round-off
    agreement plus the structural flag is the real claim."""
    spec = jno.solve.staggered([object(), object()], over_relax=1.0)
    assert spec.wants_project is False, "omega=1 must not ask for the box projector — it never steps past"

    fem, a, b = _convex_pair()
    base = np.asarray(fem.solve(nonlinear=jno.solve.staggered([a, b], rtol=1e-11, atol=1e-13)))
    one = np.asarray(fem.solve(nonlinear=jno.solve.staggered([a, b], rtol=1e-11, atol=1e-13, over_relax=1.0)))
    assert np.abs(base).max() > 1e-3
    # 1e-10, not round-off: two IDENTICAL solves differ by ~1e-17 per call on GPU and the sweep amplifies
    # that, so a bitwise-adjacent bound is flaky (it was 1e-14 and failed intermittently). This still
    # discriminates — a genuinely different iteration would differ at the SOLVE tolerance, ~1e-8.
    assert np.abs(base - one).max() <= 1e-10 * np.abs(base).max(), "over_relax=1 perturbed the default iteration"


@pytest.mark.parametrize("omega", [0.7, 1.2, 1.4, 1.8])
def test_over_relax_finds_the_same_root(omega):
    fem, a, b = _convex_pair()
    ref = np.asarray(fem.solve(nonlinear=jno.solve.staggered([a, b], rtol=1e-11, atol=1e-13)))
    got = np.asarray(fem.solve(nonlinear=jno.solve.staggered([a, b], rtol=1e-11, atol=1e-13, over_relax=omega)))
    assert np.abs(ref).max() > 1e-3, "the solution is trivial — the comparison would be vacuous"
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-8, f"omega={omega} moved the root"


@pytest.mark.parametrize("omega", [0.0, -0.5, 2.0, 2.5])
def test_over_relax_outside_kahans_range_fails_loud(omega):
    """Kahan: omega in (0, 2) is NECESSARY for the over-relaxed Gauss-Seidel iteration to converge.
    Outside it, refuse rather than iterate to a step cap and hand back a plausible non-root."""
    fem, a, b = _convex_pair(size=0.3)
    with pytest.raises(ValueError, match=r"over_relax must lie in \(0, 2\)"):
        jno.solve.staggered([a, b], over_relax=omega)


def test_over_relax_keeps_a_box_constrained_field_feasible():
    """Over-relaxation steps PAST the sub-solve's answer. The sub-solve's answer is in the box; the
    extrapolation need not be, so the driver asks the `bounds` wrapper for its projector. Without that
    the damage would leave [0, 1] mid-iteration."""
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    d.tag("left", lambda x, y: x < 1e-9)
    co, cl = d.variable("interior", split=True), d.variable("left", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    dm, q = d.fem_symbols()
    fem = jno.fem(
        [
            inner(grad(u, X), grad(phi, X), 1) + 10.0 * (u - 5.0) * phi,
            # A driving term that is NOT (1-dm)-degraded, so the unconstrained root sits at
            # dm = 6/2.5 = 2.4 and the cap genuinely binds. (An AT2 `-2(1-dm)H q` source asymptotes to
            # dm = 2H/(gc/l + 2H) < 1 however hard it is driven, so the bound would never activate.)
            2.5 * dm * q + 0.4 * inner(grad(dm, X), grad(q, X), 1) - 6.0 * q,
            dm.bounds(0.0, 1.0),
            u(*cl) - 0.0,
        ]
    )
    sol = np.asarray(fem.solve(nonlinear=jno.solve.staggered([u, dm], rtol=1e-9, atol=1e-11, over_relax=1.6)))
    dmg = sol[fem.blocks[fem.block_index(dm)]]
    assert dmg.min() >= -1e-9 and dmg.max() <= 1.0 + 1e-9, f"damage left the box: [{dmg.min()}, {dmg.max()}]"
    assert dmg.max() > 1.0 - 1e-6, "the bound must actually be ACTIVE, or this tests nothing"


def test_over_relax_does_not_move_a_dirichlet_dof():
    """Over-relaxation must act on the FREE dofs only.

    Farrell & Maurini's ``u~`` lives in the constrained space ``C_u``, so a prescribed dof has
    ``delta = 0`` and omega never reaches it. jNO imposes essential conditions as residual ROWS, so the
    sub-solve lands exactly ON the prescribed value and extrapolating past it is simply wrong. Measured
    on one row with g = 2 before the fix: omega=1.7 gave 3.40 -> 1.02 -> 2.69, an oscillation decaying
    only as |1-omega|^k, with every other field meanwhile solved against that wrong boundary value.
    """
    from jno.utils.solver.newton_krylov import staggered_newton

    g = 2.0
    A = jnp.array([[3.0, 0.4], [0.4, 2.0]])
    B = jnp.array([[2.5, -0.3], [-0.3, 4.0]])
    C = jnp.array([[0.5, 0.2], [0.1, 0.6]])

    def R(z):
        x, y = z[:2], z[2:]
        r0 = (A @ x + C @ y + 0.2 * x**3 - jnp.array([1.0, -0.5])).at[0].set(x[0] - g)
        return jnp.concatenate([r0, B @ y + C.T @ x + 0.15 * y**3 - jnp.array([0.3, 0.8])])

    blocks = [np.array([0, 1]), np.array([2, 3])]
    kw = dict(max_sweeps=3, rtol=0.0, atol=0.0, inner_steps=60, inner_tol=1e-14)
    import jno.utils.solver.newton_krylov as _nk

    keep = _nk._convergence_check
    _nk._convergence_check = lambda f0, u0, u, **k: u  # truncating on purpose; the guard is not under test
    try:
        for omega in (1.3, 1.7):
            got = np.asarray(staggered_newton(R, jnp.zeros(4), blocks, over_relax=omega, constrained=[0], **kw))
            assert abs(got[0] - g) < 1e-12, f"omega={omega} moved a prescribed dof to {got[0]}"
            loose = np.asarray(staggered_newton(R, jnp.zeros(4), blocks, over_relax=omega, **kw))
            assert abs(loose[0] - g) > 1e-3, "without the mask the dof should overshoot — else this is vacuous"
    finally:
        _nk._convergence_check = keep


def test_over_relax_on_a_ramped_dirichlet_march_matches_the_unrelaxed_answer():
    """End to end through `fem.solve`: the constrained dofs come off the operator, so a RAMPED grip
    (the case that exposed this — g changes every load step, so `g - u_prev` is large at sweep 1) is
    held exactly and the over-relaxed march lands on the same trajectory."""
    grad, inner = _aliases()
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain(tau=(0.0, 1.0, 4))
    d.tag("bot", lambda x, y: y < 1e-9)
    d.tag("top", lambda x, y: y > 1 - 1e-9)
    co, cb, ct = (d.variable(r, split=True) for r in ("interior", "bot", "top"))
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    v, chi = d.fem_symbols()
    s, _ = d.fem_symbols()
    fem = jno.fem(
        [
            inner(grad(u, X), grad(phi, X), 1) + 0.3 * u * u * phi - 0.4 * v * phi + 0.0 * s.i(-1) * phi,
            inner(grad(v, X), grad(chi, X), 1) + v * chi - 2.0 * u * chi,
            s.evolves(u),
            u(*cb) - 0.0,
            u(*ct) - 0.7 * ct[-1],  # the RAMP: a different prescribed value at every load step
        ]
    )
    kw = dict(max_sweeps=400, rtol=1e-10, atol=1e-12)
    ref = np.asarray(fem.solve(nonlinear=jno.solve.staggered([u, v], **kw)))
    got = np.asarray(fem.solve(nonlinear=jno.solve.staggered([u, v], over_relax=1.5, **kw)))
    grip = fem.region_dofs("top", field=u)
    taus = np.asarray(d._time_points)
    assert np.abs(ref).max() > 1e-3
    assert np.abs(got - ref).max() / np.abs(ref).max() < 1e-8, "the over-relaxed march moved the answer"
    assert np.abs(got[:, grip] - (0.7 * taus)[:, None]).max() < 1e-9, "the ramped grip was not held"


# --------------------------------------------------------------------------------------------------
# Oracle 7 — the retreat. `_retreat` is the ONE place this library shortens a step: a Newton step
# retreats toward alpha = 0, an extrapolation toward omega = 1 (the sub-solve's own answer). Both are
# bisection toward a known-good fallback, and both must reject a trial point where the residual is not
# finite -- which is how an over-relaxed step into an inverted element (det F <= 0) is caught.
# --------------------------------------------------------------------------------------------------
def test_retreat_returns_the_largest_admissible_parameter():
    from jno.utils.solver.newton_krylov import _retreat

    # admissible below 1.25 only: bisecting [1, 2] gives 2 -> 1.5 -> 1.25 -> 1.125, so 1.125 is the
    # first acceptable trial and the largest this bisection can reach.
    got = float(_retreat(lambda w: w < 1.25, hi=2.0, lo=1.0, max_halvings=20, dtype=jnp.float64))
    assert abs(got - 1.125) < 1e-12, got
    # everything admissible -> the trial is taken unchanged, no cost, no drift
    assert float(_retreat(lambda w: jnp.asarray(True), hi=1.7, lo=1.0, max_halvings=20, dtype=jnp.float64)) == 1.7
    # toward zero (the Newton instantiation): halving from `hi`
    got0 = float(_retreat(lambda a: a < 0.3, hi=1.0, lo=0.0, max_halvings=20, dtype=jnp.float64))
    assert abs(got0 - 0.25) < 1e-12, got0


def test_retreat_rejects_a_non_finite_trial():
    """A NaN comparison is False, so a trial where the residual blew up shortens the step. This is the
    mechanism, not a side effect — `nan < x` being False is what makes the guard NaN-safe."""
    from jno.utils.solver.newton_krylov import _retreat

    def accept(w):  # NaN past 1.2 — a stand-in for det F <= 0 in a finite-strain form
        r = jnp.where(w > 1.2, jnp.nan, 1.0)
        return jnp.linalg.norm(r) < 10.0

    got = float(_retreat(accept, hi=1.8, lo=1.0, max_halvings=30, dtype=jnp.float64))
    assert got <= 1.2, f"retreated to {got}, which is past the cliff"
    assert got > 1.0, "it retreated all the way to the fallback when a usable step existed"


def test_retreat_bottoms_out_at_the_fallback():
    """Nothing admissible -> it lands arbitrarily close to `lo`, i.e. the un-relaxed sub-solve answer,
    so an over-relaxed sweep degrades to plain alternate minimization rather than diverging."""
    from jno.utils.solver.newton_krylov import _retreat

    got = float(_retreat(lambda w: jnp.asarray(False), hi=1.8, lo=1.0, max_halvings=30, dtype=jnp.float64))
    assert abs(got - 1.0) < 1e-6, got


def test_there_is_exactly_one_armijo_implementation():
    """Anti-drift. Three byte-identical Armijo loops existed before the retreat helper, and a fourth
    step-taker (over-relaxation) had none at all — which is precisely how the finite-strain NaN got in.
    A new driver (bfgs, anderson) must reuse `_retreat`, not fork a copy."""
    import inspect

    from jno.utils.solver import newton_krylov as nk

    src = inspect.getsource(nk)
    assert src.count("1.0 - ls_c") == 1, "the Armijo predicate is written more than once — reuse `_armijo`"
    assert src.count("jax.lax.while_loop(cond, body") >= 1
    assert src.count("def _retreat(") == 1, "there must be exactly one retreat helper"
