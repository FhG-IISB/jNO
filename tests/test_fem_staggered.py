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
