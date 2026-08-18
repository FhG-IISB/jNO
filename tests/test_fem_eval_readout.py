"""``fem.eval(term, u)`` — assemble a weak term at the solution, un-eliminated.

The readout primitive. Every solve path elimination-mutates the system it keeps (symmetric elimination
for the linear route, row replacement for Newton), which zeroes exactly the rows a reaction or flux
readout asks about — so ``fem.A``/``fem.b``/``fem.residual`` return a plausible, silent **zero** there.
``eval`` assembles the free residual instead, and the conjugate quantity on a constrained region is that
vector summed over the region's DOFs: force in mechanics, heat flux in thermal, current in EM.

Oracles are global balances, not restatements of the assembly: a reaction equals the applied load
(equilibrium), and a wall flux equals the volumetric source (conservation).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno

pytest.importorskip("basix", reason="native Lagrange assembler needs basix")

E, NU = 200.0, 0.3
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))
MU = E / (2 * (1 + NU))


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def test_heat_flux_through_the_walls_equals_the_source():
    """Conservation: for -div(grad u) = f with u pinned all round, the total wall flux must equal
    the integrated source. This is the thermal reading of the same operation as a reaction force."""
    grad, inner = jno.np.grad, jno.np.inner
    f_src = 3.0
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.08).domain()
    d.tag("walls", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()

    stiff = inner(grad(u, X), grad(phi, X), 1)  # the OPERATOR term alone
    fem = jno.fem([stiff - f_src * phi, u(*cw) - 0.0])
    sol = fem.solve()

    # The free residual of the full form is ~0 on interior rows and carries the reaction on pinned rows.
    R = np.asarray(fem.eval(stiff - f_src * phi, sol))
    wall = fem.region_dofs("walls")
    interior = np.setdiff1d(np.arange(R.size), wall)
    assert np.abs(R[interior]).max() < 1e-9, "the solved form must be in equilibrium off the pinned rows"

    # Total outflux = integrated source (domain area x f).
    assert abs(-R[wall].sum() - f_src * 1.0) < 1e-8, f"flux {-R[wall].sum():.6f} vs source {f_src:.6f}"


def test_reaction_on_a_loaded_bar_equals_the_applied_body_force():
    """Equilibrium: a bar under a uniform body force, pinned at one end. The reaction summed over the
    pinned face must be equal and opposite to the total load — the mechanics reading of the same
    operation as the wall flux above."""
    sym, grad, trace, inner = jno.np.sym, jno.np.grad, jno.np.trace, jno.np.inner
    bx, Lx, Ly = 2.5, 2.0, 1.0
    d = jno.Shape.rect(0.0, 0.0, Lx, Ly, size=0.12).domain()
    d.tag("left", lambda x, y: x < 1e-9)
    co, cl = d.variable("interior", split=True), d.variable("left", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols(value_shape=(2,))
    eps = lambda w: sym(grad(w, X))
    bvec = jno.np.asarray([bx, 0.0])
    mech = LAM * trace(eps(u)) * trace(eps(phi)) + 2 * MU * inner(eps(u), eps(phi), 2) - inner(bvec, phi, 1)

    fem = jno.fem([mech, u(*cl)[0] - 0.0, u(*cl)[1] - 0.0])
    sol = fem.solve()

    R = np.asarray(fem.eval(mech, sol))
    fx = R[fem.region_dofs("left", component=0)].sum()
    total = bx * Lx * Ly
    assert abs(fx + total) < 1e-7, f"reaction {fx:.6f} should balance the total body force {total:.6f}"


def test_the_eliminated_system_would_have_returned_zero():
    """The reason this method exists: fem.b (post-elimination) is ZERO on the pinned rows, so the
    naive readout is silently wrong rather than loudly unavailable."""
    grad, inner = jno.np.grad, jno.np.inner
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.15).domain()
    d.tag("walls", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    form = inner(grad(u, X), grad(phi, X), 1) - 3.0 * phi
    fem = jno.fem([form, u(*cw) - 0.0])
    sol = fem.solve()
    wall = fem.region_dofs("walls")

    A, b = np.asarray(fem.A.todense() if hasattr(fem.A, "todense") else fem.A), np.asarray(fem.b)
    naive = (A @ np.asarray(sol).reshape(-1) - b)[wall]
    assert np.abs(naive).max() < 1e-9  # the trap: identically zero, and wrong

    live = np.asarray(fem.eval(form, sol))[wall]
    assert np.abs(live).max() > 1e-3  # eval gives the real reaction


def test_eval_integrates_an_expression_with_no_test_function():
    """A test-free expression is not an assembly — it is an integrand, and ``eval`` reduces it to the
    scalar ``∫ F dΩ`` rather than to one value per DOF. Pinned here because this file used to assert
    the opposite (that it was refused); the integral's own oracles live in test_fem_functional.py."""
    grad, inner = jno.np.grad, jno.np.inner
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    d.tag("walls", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    fem = jno.fem([inner(grad(u, X), grad(phi, X), 1) - phi, u(*cw) - 0.0])
    sol = fem.solve()

    weak = np.asarray(fem.eval(u * u * phi, sol))  # the SAME integrand, carrying a test function
    integral = np.asarray(fem.eval(u * u, sol))
    assert weak.shape == (fem.dofs,), "a weak term still assembles to one value per DOF"
    assert integral.shape == (), f"an integrand reduces to a scalar, got shape {integral.shape}"
    assert abs(float(integral) - float(weak.sum())) < 1e-10  # partition of unity: Σ_a φ_a ≡ 1


def test_region_dofs_names_an_unknown_region():
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.3).domain()
    d.tag("walls", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    co, cw = d.variable("interior", split=True), d.variable("walls", split=True)
    X = [co[0], co[1]]
    u, phi = d.fem_symbols()
    fem = jno.fem([jno.np.inner(jno.np.grad(u, X), jno.np.grad(phi, X), 1) - phi, u(*cw) - 0.0])
    with pytest.raises(KeyError, match="unknown region"):
        fem.region_dofs("nope")
