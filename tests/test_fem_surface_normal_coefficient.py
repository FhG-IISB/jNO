"""A varying coefficient times a boundary normal in a vector surface term.

`inner(c * nr, phi, n_contract=1)` with `c` varying along the facet was wrong at the default quadrature: the per-facet
normal `(dim,)` met the coefficient `(n_q, 1)` in the quadrature-axis alignment, which read the normal's first axis as
the quadrature axis and broadcast silently whenever `n_q == dim` (2 points on a 2-D P1 edge, 3 on a 3-D P1 face).
"""

import jax
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


def _load(coefficient, **fem_kwargs):
    """The load vector of ``int_right c * n . phi ds`` on a 2 x 1 rectangle as ``(n_nodes, 2)`` (the mass form only
    gives the operator a volume term)."""
    d = jno.shape.rect(0, 0, 2, 1, size=0.25).domain()
    x, y, _ = d.variable("interior", split=True)
    xr, yr, _, _nxr, _nyr = d.variable("right", normals=True, split=True)
    nr = d.variable("right", normals=True)
    u, phi = d.fem_symbols(value_shape=(2,))
    mass = jno.np.inner(u.bind(x=x, y=y), phi.bind(x=x, y=y))
    c = {"constant": 1.0, "coordinate": yr, "linear_field": 1.0 + 2.0 * yr}[coefficient]
    load = jno.np.inner(c * nr, phi.bind(x=xr, y=yr), n_contract=1)
    _A, b = jno.fem([mass, -load], **fem_kwargs).operator
    return np.asarray(b).reshape(-1, 2), d


@pytest.mark.parametrize("coefficient, expected", [("constant", 1.0), ("coordinate", 0.5), ("linear_field", 2.0)])
def test_a_varying_coefficient_times_the_normal_integrates_exactly_at_the_default_quadrature(coefficient, expected):
    """The right edge is x = 2, y in [0, 1], n = (1, 0): the nodal loads of int c(y) n . phi sum to int_0^1 c dy in the
    x component (partition of unity) and to zero in the y component, whatever the quadrature degree."""
    b, _ = _load(coefficient)
    np.testing.assert_allclose(np.abs(b.sum(axis=0)), [expected, 0.0], atol=1e-12)
    b4, _ = _load(coefficient, quad_degree=4)
    np.testing.assert_allclose(b, b4, atol=1e-12)
