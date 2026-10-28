"""How the element kernels lower: what XLA can fuse, and what the tangent's element blocks look like.

Under ``jacfwd`` every reduction and transpose in an element kernel is a separate write of a
``(cells, tangents, points, ...)`` array, so these pin the lowering, not only the values (which the FEM
suites check): a small matrix product has no reduction, a vector field's gradient no transposes, and
the tangent's element blocks are bitwise symmetric where the form is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno.utils.solver.small_linalg import small_matmul


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


def test_a_small_matrix_product_has_no_reduction():
    # Break: lowering the k-sum as broadcast-multiply + reduce_sum (a fusion boundary under jacfwd).
    a = jnp.ones((7, 3, 3))
    assert "reduce_sum" not in _prims(small_matmul, a, a)
    assert "dot_general" not in _prims(small_matmul, a, a)


def test_a_vector_gradient_has_no_transpose():
    # Break: building it one direction at a time and moving the component axis to the end and back.
    from jno.utils.solver.fem_utils import _unpack_components

    class _Field:
        value_shape = (3,)

    flat = jnp.ones((4, 3, 3))  # (points, components, directions)
    out = _unpack_components(flat, _Field(), axis=1)
    assert out.shape == (4, 3, 3)
    assert "transpose" not in _prims(lambda f: _unpack_components(f, _Field(), axis=1), flat)


def test_the_tangent_of_a_variational_vector_form_is_bitwise_symmetric_where_assembly_is_deterministic():
    # On the CPU the global sum is deterministic, so a bitwise symmetric element block gives a bitwise
    # symmetric matrix. Break: dropping the element-block average (the forward-over-reverse block is
    # symmetric only to ~5e-17).
    sys.path.insert(0, str(Path(__file__).parent))
    import test_fem_dirichlet_symmetry as T

    with jax.default_device(jax.devices("cpu")[0]):
        fem, _d = T._elasticity()
        A = T._dense_tangent(fem, np.asarray(fem.solve()))
    assert np.array_equal(A, A.T), f"asymmetry {T._asymmetry(A):.2e}"
