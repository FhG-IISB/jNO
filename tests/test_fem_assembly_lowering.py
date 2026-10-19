"""How the element kernels lower: what XLA can fuse, and what the tangent's element blocks look like.

Under ``jacfwd`` every reduction and transpose in an element kernel is a separate write of a
``(cells, tangents, points, ...)`` array, so these pin the lowering, not only the values (which the FEM
suites check): a small matrix product has no reduction, a vector field's gradient no transposes, and
the tangent's element blocks are bitwise symmetric where the form is.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _prims(fn, *args):
    return str(jax.make_jaxpr(fn)(*args))


def test_a_vector_gradient_has_no_transpose():
    # Break: building it one direction at a time and moving the component axis to the end and back.
    from jno.utils.solver.fem_utils import _unpack_components

    class _Field:
        value_shape = (3,)

    flat = jnp.ones((4, 3, 3))  # (points, components, directions)
    out = _unpack_components(flat, _Field(), axis=1)
    assert out.shape == (4, 3, 3)
    assert "transpose" not in _prims(lambda f: _unpack_components(f, _Field(), axis=1), flat)
