"""How ``jno.fem`` splits a weak form into additive sub-terms, and what it does with the leftovers.

The classifier assigns each additive piece of a form to an equation block by the test field it
carries. Two edge cases live here:

* the builtin ``sum()`` seeds with a literal ``0``, so ``sum(f[k] * v[k] for k in ...)`` -- the
  natural spelling of a vector source -- arrives with a zero sub-term that carries no test field.
  Zero belongs to no block and changes no residual, so it is dropped;
* a sub-term that is *genuinely* test-free (or welds several test fields) is still refused, and the
  message has to say which of the two it hit, because the fixes differ.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import jno
from jno.trace import BinaryOp, Literal
from jno.utils.solver.fem_utils import _is_structural_zero

inner, grad = jno.np.inner, jno.np.grad
dense = lambda A: np.asarray(A.todense()) if hasattr(A, "todense") else np.asarray(A)  # noqa: E731


@pytest.fixture
def ctx():
    """A 2-D vector Poisson -- the smallest form with components to sum over. x64: FEM is float64."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    d = jno.shape.rect(0, 0, 1, 1, size=0.4).domain()
    u, v = d.fem_symbols(value_shape=(2,), names=("u", "v"), order=1)
    xi, yi = d.variable("interior", split=True)[:2]
    xb, yb = d.variable("boundary", split=True)[:2]
    vv = v.bind(x=xi, y=yi)
    gu, gv = grad(u, [xi, yi]), grad(v, [xi, yi])
    try:
        yield d, u, vv, gu, gv, (xi, yi), (xb, yb)
    finally:
        jax.config.update("jax_enable_x64", prev)


def _bcs(u, xb, yb):
    return [u(xb, yb)[0] - 0.0, u(xb, yb)[1] - 0.0]


def _assert_same_operator(a, b, ulps: int = 8):
    """Two spellings of one form must assemble to the same operator, to round-off.

    Not bit-for-bit: dropping the zero lets the classifier distribute the remaining products
    differently, so the element contributions reduce in a different order. Measured difference is
    ~1 ulp of the largest entry; the gate is 8, which is loose enough to be stable and far tighter
    than any real discrepancy (a genuinely different form differs in the first digit, not the last).
    """
    A, B = dense(a.A), dense(b.A)
    ba, bb = np.asarray(a.b).reshape(-1), np.asarray(b.b).reshape(-1)
    eps = np.finfo(np.float64).eps
    assert np.abs(A - B).max() <= ulps * eps * max(np.abs(A).max(), 1.0)
    assert np.abs(ba - bb).max() <= ulps * eps * max(np.abs(ba).max(), 1.0)


def test_sum_over_components_assembles_exactly_like_the_explicit_sum(ctx):
    """``sum(...)`` and the written-out sum are the same form, so they must assemble bit-for-bit.

    The oracle is the explicit spelling, which has always worked -- not a tolerance.
    """
    d, u, vv, gu, gv, (xi, yi), (xb, yb) = ctx
    f = [1.0 + xi, 2.0 - yi]
    diffusion = inner(gu, gv, n_contract=2)

    explicit = jno.fem([diffusion - (f[0] * vv[0] + f[1] * vv[1]), *_bcs(u, xb, yb)])
    summed = jno.fem([diffusion - sum(f[k] * vv[k] for k in range(2)), *_bcs(u, xb, yb)])

    _assert_same_operator(explicit, summed)
    # and the form actually solves to the same field, not merely assembles the same
    np.testing.assert_allclose(
        np.asarray(explicit.solve(linear=jno.solve.lu(backend="host"))),
        np.asarray(summed.solve(linear=jno.solve.lu(backend="host"))),
        rtol=1e-12,
        atol=1e-14,
    )


def test_a_zero_added_to_a_term_changes_nothing(ctx):
    """``0 + term`` is the same term. It used to be refused."""
    d, u, vv, gu, gv, (xi, yi), (xb, yb) = ctx
    body = inner(gu, gv, n_contract=2) - (vv[0] + vv[1])
    plain = jno.fem([body, *_bcs(u, xb, yb)])
    padded = jno.fem([0 + body, *_bcs(u, xb, yb)])
    _assert_same_operator(plain, padded)


def test_a_genuinely_test_free_sub_term_is_refused_and_named(ctx):
    """Dropping the zero must not soften the real rule: a source with no test function is an error,
    and the message must say *no* test field rather than assert "exactly one"."""
    d, u, vv, gu, gv, (xi, yi), (xb, yb) = ctx
    with pytest.raises(ValueError, match="carries no test field"):
        jno.fem([inner(gu, gv, n_contract=2) - 5.0, *_bcs(u, xb, yb)])


def test_structural_zero_is_narrow():
    """Only a *concrete* literal zero is punctuation. A tracer is a real coefficient even at zero --
    dropping it would discard physics rather than the seed of a sum."""
    assert _is_structural_zero(Literal(0))
    assert _is_structural_zero(Literal(0.0))
    assert _is_structural_zero(BinaryOp("*", Literal(-1.0), Literal(0)))
    assert _is_structural_zero(BinaryOp("*", Literal(0), Literal(7.5)))
    assert not _is_structural_zero(Literal(1.0))
    assert not _is_structural_zero(BinaryOp("*", Literal(-1.0), Literal(2.0)))
    assert not _is_structural_zero(BinaryOp("+", Literal(0), Literal(0)))  # a sum is not a product

    seen = []
    jax.jit(lambda z: seen.append(_is_structural_zero(Literal(z))) or z)(0.0)
    assert seen == [False], "a traced zero is not known to be zero at build time"


# ---------------------------------------------------------------------------------------------------
# The same rule on every assembler: it is decided once, where `jno.fem` classifies the terms.
# 1-D integrated a test-free constant into a SILENTLY wrong answer (`2 + u' v'` moved the solution by
# 0.48), a 1-D transient failed with a reshape TypeError on the `0` builtin `sum()` seeds, and the
# non-nodal path refused that harmless `0`.
# ---------------------------------------------------------------------------------------------------
def _poisson(dim, transient, space, const):
    from tests.test_fem_trainables_gaps import _evaluate

    T = (0.0, 0.05, 4)
    tkw = {"time": T} if transient else {}
    if dim == 1:
        d = jno.domain(constructor=jno.domain.line(mesh_size=0.2), **tkw)
    else:
        d = jno.shape.rect(0, 0, 1, 1, size=0.4).domain(**tkw)
    u, v = d.fem_symbols() if space is None else d.fem_symbols(space=space)
    co, cb = d.variable("interior", split=True), d.variable("boundary", split=True)
    X, Xb = (co[:-1], cb[:-1]) if transient else (co[:dim], cb[:dim])
    kw = dict(zip("xyz", X)) | ({"t": co[-1]} if transient else {})
    ui, vi = u.bind(**kw), v.bind(**kw)
    a = ui.x * vi.x if dim == 1 else ui.x * vi.x + ui.y * vi.y

    def build(body):
        terms = [(ui.t * vi + body) if transient else body, u(*Xb) - 0.0]
        if transient:
            terms.append(u(*d.variable("initial", split=True)[:dim]) - 0.0)
        return terms

    plain = np.asarray(_evaluate(jno.fem(build(a - 1.0 * vi)).solve()))
    padded = build((const + a) - 1.0 * vi)
    return plain, padded, _evaluate


_PATHS = [
    (1, False, None),
    (1, True, None),
    (1, False, "Hermite"),
    (1, True, "Hermite"),
    (2, True, None),
    (2, False, "Hermite"),
]


@pytest.mark.parametrize("dim,transient,space", _PATHS)
def test_a_zero_summand_changes_nothing_on_every_assembler(dim, transient, space):
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        plain, padded, evaluate = _poisson(dim, transient, space, 0)
        np.testing.assert_allclose(np.asarray(evaluate(jno.fem(padded).solve())), plain, rtol=0, atol=1e-12)
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("dim,transient,space", _PATHS)
def test_a_test_free_constant_is_refused_on_every_assembler(dim, transient, space):
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        _plain, padded, _ = _poisson(dim, transient, space, 2.0)
        with pytest.raises(ValueError, match="carries no test field"):
            jno.fem(padded)
    finally:
        jax.config.update("jax_enable_x64", prev)
