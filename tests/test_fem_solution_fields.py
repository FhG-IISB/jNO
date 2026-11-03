"""``fem.solve()`` returns a :class:`jno.Solution`: the fields by name (``u.T``, ``u.A``) and still the flat vector
of every DOF (indexing, arithmetic, ``jnp``/``np`` functions, ``x0=``, a pytree under ``jit``/``grad``).

Oracles: the field blocks against slices of the flat vector at the problem's own offsets, and a two-field problem
whose second field is known in closed form (S = 2 everywhere, pinned).
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


def _two_fields(**solve):
    """T: -lap T = S on the unit square, T = 0 on the boundary; S = 2 (its own equation S - 2 = 0, as a mass form)."""
    d = jno.shape.rect(0, 0, 1, 1).sized(0.2).domain()
    T, S = d.unknown.scalar("T"), d.unknown.scalar("S")
    v, q = T.test(), S.test()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    Ti, vi, Si, qi = T.bind(x=x, y=y), v.bind(x=x, y=y), S.bind(x=x, y=y), q.bind(x=x, y=y)
    fem = jno.fem([Ti.d(x) * vi.d(x) + Ti.d(y) * vi.d(y) - Si * vi, (Si - 2.0) * qi, T(xb, yb) - 0.0])
    return fem, fem.solve(linear=jno.solve.lu(), **solve)


def test_the_fields_are_read_by_name_and_are_the_blocks_of_the_flat_vector():
    fem, u = _two_fields()
    n = int(fem.offsets[-1])
    flat = np.asarray(u)
    assert isinstance(u, jno.Solution) and flat.shape == (n,) and set(u.fields) == {"T", "S"}
    i, j = list(fem.offsets[:2]), list(fem.offsets[1:3])
    blocks = {"T": flat[i[0] : j[0]], "S": flat[i[1] : j[1]]}
    for name in ("T", "S"):
        assert np.array_equal(np.asarray(getattr(u, name)), blocks[name])
        assert np.array_equal(np.asarray(u[name]), blocks[name])
    assert np.allclose(np.asarray(u.S), 2.0) and np.asarray(u.T).max() > 0


def test_a_field_named_T_is_the_field_not_the_transpose():
    _, u = _two_fields()
    assert np.asarray(u.T).shape != np.asarray(u).shape


def test_it_still_works_as_the_flat_array():
    fem, u = _two_fields()
    flat = np.asarray(u)
    assert u.shape == flat.shape and u.dtype == flat.dtype and len(u) == flat.size
    assert float(u[0]) == flat[0] and np.allclose(np.asarray(u[1:4]), flat[1:4])
    assert np.allclose(np.asarray(2 * u - u), flat) and np.allclose(np.asarray(u @ u), flat @ flat)
    assert float(jnp.linalg.norm(u)) == pytest.approx(np.linalg.norm(flat))
    assert np.allclose(np.asarray(jnp.asarray(u)), flat)
    _, again = _two_fields(x0=u)  # a solution passed back in as a start
    assert np.allclose(np.asarray(again), flat)


def test_it_is_a_pytree_of_the_flat_vector():
    fem, u = _two_fields()
    (leaf,) = jax.tree_util.tree_leaves(u)
    assert np.array_equal(np.asarray(leaf), np.asarray(u))
    doubled = jax.jit(lambda s: jax.tree_util.tree_map(lambda a: 2 * a, s))(u)
    assert isinstance(doubled, jno.Solution) and np.allclose(np.asarray(doubled.S), 4.0)


def test_an_unknown_field_name_says_which_fields_there_are():
    _, u = _two_fields()
    with pytest.raises(AttributeError, match=r"'S', 'T'"):
        u.pressure


def test_a_complex_field_comes_back_complex_and_a_constant_as_its_value():
    """A complex field (two real halves) is one complex array; a constant unknown its single value."""
    d = jno.shape.rect(0, 0, 1, 1).sized(0.25).domain()
    A = d.unknown.scalar("A", complex=True)
    I = d.unknown.scalar("I", constant=True)  # noqa: E741
    W = A.test()
    x, y, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    Ai, Wi = A.bind(x=x, y=y), W.bind(x=x, y=y)
    fem = jno.fem([Ai.d(x) * Wi.d(x) + Ai.d(y) * Wi.d(y) + 1j * Ai * Wi - I * Wi, A(xb, yb) - 0.0, I - 3.0])
    u = fem.solve(linear=jno.solve.lu())
    assert np.iscomplexobj(np.asarray(u.A)) and np.abs(np.asarray(u.A).imag).max() > 0
    assert np.asarray(u.I).shape == () and float(u.I) == pytest.approx(3.0)
