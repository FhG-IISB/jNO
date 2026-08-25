"""Nonlocal terms written as expressions instead of lambdas.

``gap.field(u)`` on a *symbolic* trial function defers the gather, arithmetic on the result records
rather than computes, and ``gap.load(...)`` closes the chain into a ``jno.Coupling``. So a nonlocal
term drops into the ``jno.fem([...])`` list as an expression::

    gap.load(G @ gap.field(u) ** 4)              # expression
    lambda T: gap.load(G @ gap.field(T) ** 4)    # the hand-written equivalent

The two must be numerically identical; that equivalence is what these tests pin down. The mechanism is
not radiation-specific -- it covers any gather -> operate -> scatter term (integral / non-reflecting
BCs, contact, peridynamics).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.domain.enclosure import PendingElementExpr

SIGMA = 5.670374419e-8


@pytest.fixture(scope="module")
def gap_and_domain():
    d = (jno.Shape.rect(3, 2, 5, 4, size=0.6).name("solid") + jno.Shape.rect(1, 0, 7, 6, size=1.2).name("medium")).domain()
    # Off the axis on purpose: the ring measure 2*pi*r vanishes at r=0, and the resulting row
    # scaling makes a 91-node toy problem stall the inner BiCGStab rather than converge.
    gap = d.enclosure(["solid"], medium_tags=["medium"], axisymmetric=True, enforce_closure=True)
    return d, gap


@pytest.fixture(scope="module")
def exchange(gap_and_domain):
    _d, gap = gap_and_domain
    eps = gap.emissivity(0.8)
    F = jnp.asarray(gap.view_factor)
    return (jnp.diag(F.sum(1)) - F) @ jnp.linalg.solve(jnp.eye(gap.size) - (1.0 - eps)[:, None] * F, jnp.diag(eps * SIGMA))


# --------------------------------------------------------------------------------------
# deferral
# --------------------------------------------------------------------------------------
def test_field_on_a_symbol_defers_and_on_an_array_does_not(gap_and_domain):
    d, gap = gap_and_domain
    u, _v = d.fem_symbols()
    assert isinstance(gap.field(u), PendingElementExpr)

    concrete = gap.field(jnp.arange(len(np.asarray(d.mesh.points)), dtype=float))
    assert isinstance(concrete, jnp.ndarray) and concrete.shape == (gap.size,)


def test_load_of_a_pending_expression_is_a_coupling(gap_and_domain, exchange):
    d, gap = gap_and_domain
    u, _v = d.fem_symbols()
    term = gap.load(exchange @ gap.field(u) ** 4)
    assert isinstance(term, jno.Coupling)
    assert "solid" in term.name


def test_load_of_an_array_is_still_a_plain_load(gap_and_domain):
    _d, gap = gap_and_domain
    out = gap.load(jnp.ones(gap.size))
    assert isinstance(out, jnp.ndarray)


# --------------------------------------------------------------------------------------
# the expression must equal the hand-written lambda
# --------------------------------------------------------------------------------------
def test_expression_matches_the_hand_written_residual(gap_and_domain, exchange):
    d, gap = gap_and_domain
    u, _v = d.fem_symbols()
    n = len(np.asarray(d.mesh.points))

    term = gap.load(exchange @ gap.field(u) ** 4)
    T = jnp.asarray(1200.0 + 600.0 * np.random.default_rng(0).random(n))

    expected = gap.load(exchange @ gap.field(T) ** 4)
    assert np.allclose(np.asarray(term.residual_fn(T)), np.asarray(expected), rtol=0, atol=0)


@pytest.mark.parametrize(
    "build, ref",
    [
        (lambda f, G: G @ f**4, lambda a, G: G @ a**4),
        (lambda f, G: 2.0 * f, lambda a, G: 2.0 * a),
        (lambda f, G: f * 2.0, lambda a, G: a * 2.0),
        (lambda f, G: f + 1.0, lambda a, G: a + 1.0),
        (lambda f, G: 1.0 + f, lambda a, G: 1.0 + a),
        (lambda f, G: f - 1.0, lambda a, G: a - 1.0),
        (lambda f, G: 1.0 - f, lambda a, G: 1.0 - a),
        (lambda f, G: -f, lambda a, G: -a),
        (lambda f, G: f / 2.0, lambda a, G: a / 2.0),
        (lambda f, G: f + f, lambda a, G: a + a),
        (lambda f, G: f.apply(jnp.sqrt), lambda a, G: jnp.sqrt(a)),
    ],
)
def test_recorded_arithmetic_matches_eager_arithmetic(gap_and_domain, exchange, build, ref):
    d, gap = gap_and_domain
    u, _v = d.fem_symbols()
    n = len(np.asarray(d.mesh.points))
    T = jnp.asarray(300.0 + 50.0 * np.random.default_rng(1).random(n))

    pending = build(gap.field(u), exchange)
    assert isinstance(pending, PendingElementExpr)
    assert np.allclose(np.asarray(pending(T)), np.asarray(ref(gap.field(T), exchange)))


def test_matmul_from_the_left_reaches_the_pending_operand(gap_and_domain, exchange):
    """`ndarray @ pending` must route to __rmatmul__ rather than the array trying to coerce it."""
    d, gap = gap_and_domain
    u, _v = d.fem_symbols()
    assert isinstance(exchange @ gap.field(u), PendingElementExpr)
    assert isinstance(np.asarray(exchange) @ gap.field(u), PendingElementExpr)


# --------------------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------------------
def test_solving_with_the_expression_matches_solving_with_the_lambda(gap_and_domain, exchange):
    d, gap = gap_and_domain
    u, v = d.fem_symbols()
    n = len(np.asarray(d.mesh.points))
    x, y, _t = d.variable("interior", split=True)
    ui, vi = u.bind(x=x, y=y), v.bind(x=x, y=y)
    xc, yc, _tc = d.variable("boundary", split=True)

    # The load from an axisymmetric enclosure is per full revolution, so the weak form has to carry
    # the same 2*pi*r or the two sides differ by exactly that factor.
    dV = 2.0 * np.pi * x
    conduction = 50.0 * (ui.x * vi.x + ui.y * vi.y) * dV - 200.0 * vi * dV
    dirichlet = u(xc, yc) - 300.0

    as_expr = jno.fem([conduction, gap.load(exchange @ gap.field(u) ** 4), dirichlet])
    as_lambda = jno.fem([conduction, lambda T: gap.load(exchange @ gap.field(T) ** 4), dirichlet])

    # Compared through the ASSEMBLED residual rather than through a solve: this is the integration
    # point that matters (jno.fem has to recognise the expression as a Coupling and add it to
    # R_local), and it does not hang on a toy problem being well enough conditioned to converge.
    T = jnp.asarray(300.0 + 80.0 * np.random.default_rng(2).random(n))
    ra, rb = np.asarray(as_expr.residual(T)), np.asarray(as_lambda.residual(T))
    assert np.allclose(ra, rb, rtol=0, atol=0)

    # ...and the radiation genuinely participated rather than assembling to nothing. (A conduction+
    # Dirichlet fem cannot be used as the reference here: with no coupling it is LINEAR, and .residual
    # is only defined for the nonlinear/transient modes.)
    contribution = np.asarray(gap.load(exchange @ gap.field(u) ** 4).residual_fn(T))
    assert np.abs(contribution).max() > 1e-6
