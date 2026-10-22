"""``jno.np.dot`` in a weak form is the per-point tensor product, chosen by the operands' ranks.

Inside an element kernel every operand carries leading batch axes (the quadrature axis, and a local-basis
axis on a test function) with the tensor in the TRAILING axes. ``jnp.dot`` contracted across those batch
axes, so ``dot(F, F)`` of two ``(n_q, 3, 3)`` operands came out ``(n_q, 3, n_q, 3)`` and the form died on a
shape error. A shape alone cannot say whether ``(4, 3)`` is four vectors or one 4x3 matrix, so ``dot``
reads the rank from the trace (a field's ``value_shape``, ``grad`` adding one, views, constants) and refuses
when it cannot.

Oracle: the same physics written with ``einsum`` / ``inner``, which already contract trailing axes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _residual(form):
    """Residual of ``form(u, ub, phib, F, s)`` + a clamped face, at a fixed random state."""
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    x, y, z = d.variable("interior", split=True)[:3]
    xl, yl, zl = d.variable("left", split=True)[:3]
    # P2: F varies inside a cell and the basis count (10) differs from the quadrature count, so a product
    # that mixes batch axes cannot agree with the reference by coincidence (it did with P1: 4 and 4).
    u, phi = d.fem_symbols(value_shape=(3,), order=2)
    X = [x, y, z]
    ub, phib = u.bind(x=x, y=y, z=z), phi.bind(x=x, y=y, z=z)
    F = jno.np.identity(3) + jno.np.grad(u, X)
    fem = jno.fem([form(u, ub, phib, phi, F, X), u(xl, yl, zl) - (0.0, 0.0, 0.0)])
    u0 = jnp.asarray(np.random.default_rng(0).standard_normal(fem.dofs) * 0.1)
    return np.asarray(fem.residual(u0))


dot, inner, einsum, grad, T = jno.np.dot, jno.np.inner, jno.np.einsum, jno.np.grad, jno.np.transpose

CASES = {
    # matrix . matrix: Green-Lagrange strain from F^T F
    "matrix-matrix": (
        lambda u, ub, pb, p, F, X: inner(0.5 * dot(T(F), F), grad(p, X), n_contract=2),
        lambda u, ub, pb, p, F, X: inner(0.5 * einsum("...ki,...kj->...ij", F, F), grad(p, X), n_contract=2),
    ),
    # matrix . vector: a pushed-forward vector against the test function
    "matrix-vector": (
        lambda u, ub, pb, p, F, X: inner(dot(F, ub), pb),
        lambda u, ub, pb, p, F, X: inner(einsum("...ij,...j->...i", F, ub), pb),
    ),
    # vector . matrix
    "vector-matrix": (
        lambda u, ub, pb, p, F, X: inner(dot(ub, F), pb),
        lambda u, ub, pb, p, F, X: inner(einsum("...i,...ij->...j", ub, F), pb),
    ),
    # vector . vector -> a scalar coefficient
    "vector-vector": (
        lambda u, ub, pb, p, F, X: (1.0 + dot(ub, ub)) * inner(grad(u, X), grad(p, X), n_contract=2),
        lambda u, ub, pb, p, F, X: (1.0 + inner(ub, ub)) * inner(grad(u, X), grad(p, X), n_contract=2),
    ),
    # the TEST function inside dot: it carries a local-basis axis the other operand does not.
    # (F phi) . u == phi . (F^T u)
    "matrix-testfunction": (
        lambda u, ub, pb, p, F, X: inner(dot(F, pb), ub),
        lambda u, ub, pb, p, F, X: inner(pb, einsum("...ji,...j->...i", F, ub)),
    ),
    # scalar . vector is a scaling, as in numpy
    "scalar-vector": (
        lambda u, ub, pb, p, F, X: inner(dot(2.0, ub), pb),
        lambda u, ub, pb, p, F, X: inner(2.0 * ub, pb),
    ),
}


@pytest.mark.parametrize("case", list(CASES))
def test_dot_in_a_weak_form_is_the_per_point_product(case):
    # Break: jnp.dot (contracts across the batch axes), plain matmul (reads a vector as a matrix), or a
    # missing basis-axis alignment for the test function.
    with_dot, reference = CASES[case]
    r, r_ref = _residual(with_dot), _residual(reference)
    assert np.abs(r_ref).max() > 1e-3  # a vacuous all-zero residual would compare equal
    np.testing.assert_allclose(r, r_ref, rtol=1e-12, atol=1e-13 * np.abs(r_ref).max())  # summation order


def test_dot_refuses_an_operand_whose_rank_it_cannot_tell():
    # Break: guessing the rank of an opaque expression from its runtime shape.
    d = jno.shape.box(0, 0, 0, 1, 1, 1, size=0.5).domain()
    x, y, z = d.variable("interior", split=True)[:3]
    u, _ = d.fem_symbols(value_shape=(3,))
    opaque = jno.np.function(lambda a: a, [u]) if hasattr(jno.np, "function") else None
    with pytest.raises(ValueError, match="inner.*matmul|matmul.*inner"):
        dot(opaque, u)


def test_dot_of_plain_arrays_is_numpy_dot():
    # Outside a trace there are no batch axes: numpy semantics, including N-D.
    rng = np.random.default_rng(1)
    for sa, sb in [((3,), (3,)), ((2, 3), (3,)), ((2, 3), (3, 4)), ((2, 3, 4), (5, 4, 2))]:
        a, b = rng.standard_normal(sa), rng.standard_normal(sb)
        got = dot(jnp.asarray(a), jnp.asarray(b))
        got = got.fn(*[arg.value if hasattr(arg, "value") else arg for arg in got.args]) if hasattr(got, "fn") else got
        np.testing.assert_allclose(np.asarray(got), np.dot(a, b), rtol=1e-13, atol=1e-13)
