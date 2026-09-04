"""Symmetric Dirichlet elimination scales the pinned row to the matrix, instead of pinning it to 1.

``_apply_dirichlet_symmetric`` is the single Dirichlet path for every sparse assembler in jNO --
``fem_native`` (nodal), ``fem_nonnodal`` (N1E / RT / P0 / Hermite / Argyris / Morley) and
``fem_1d`` -- so this is the widest-reach numerical change in the FEM stack and gets pinned here.

Setting the eliminated row to ``e_i`` (a unit diagonal) is algebraically correct and numerically
awful: a stiffness matrix whose entries are O(1e6) ends up with a handful of O(1) rows, and the
condition number picks up the whole ratio. Scaling the row by the magnitude of that DOF's ORIGINAL
diagonal -- row ``s_i * e_i``, load ``s_i * g_i`` -- leaves the solution identical (``u_i = g_i``
either way) and keeps the row on the same scale as the rest of the matrix.

The measured effect on the 1-D Laplacian below is cond 5.8 against 3.4e6.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.experimental import sparse as jsp  # noqa: E402

from jno.utils.solver.fem_1d import _apply_dirichlet_symmetric  # noqa: E402

PINS = [(0, 0.5), (4, -0.25)]


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _laplacian(n=5, scale=1e6):
    K = np.diag(2.0 * np.ones(n)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
    return K * scale


def _eliminated(K, pins=PINS, rhs=None):
    n = K.shape[0]
    b = jnp.asarray(np.full(n, K.max()) if rhs is None else rhs)
    A2, b2 = _apply_dirichlet_symmetric(jsp.BCOO.fromdense(jnp.asarray(K)), b, pins)
    return np.asarray(A2.todense()), np.asarray(b2)


def test_the_dirichlet_values_are_still_enforced_exactly():
    """The scaling must be solution-preserving: row ``s*e_i`` with load ``s*g`` still gives u_i = g."""
    K = _laplacian()
    D, b = _eliminated(K)
    u = np.linalg.solve(D, b)
    for dof, g in PINS:
        assert np.isclose(u[dof], g, rtol=0, atol=1e-12), f"dof {dof}: {u[dof]} != {g}"


def test_the_pinned_row_is_scaled_to_the_matrix_not_to_one():
    K = _laplacian()
    D, _ = _eliminated(K)
    for dof, _g in PINS:
        assert np.isclose(D[dof, dof], abs(K[dof, dof]), rtol=1e-12), (
            f"dof {dof} diagonal is {D[dof, dof]:.3e}, expected the original |A_ii| "
            f"{abs(K[dof, dof]):.3e} (1.0 means the scaling was dropped)"
        )


def test_symmetry_survives_the_scaling():
    """Only the diagonal is touched, so a symmetric operator must stay symmetric -- CG depends on it."""
    D, _ = _eliminated(_laplacian())
    assert np.allclose(D, D.T, rtol=0, atol=0)


def test_conditioning_beats_the_unit_diagonal():
    K = _laplacian()
    D, _ = _eliminated(K)
    unit = D.copy()
    for dof, _g in PINS:
        unit[dof, dof] = 1.0
    scaled_cond, unit_cond = np.linalg.cond(D), np.linalg.cond(unit)
    assert scaled_cond < unit_cond / 1e3, f"scaling bought nothing: cond {scaled_cond:.3e} vs unit-diagonal {unit_cond:.3e}"


def test_a_dof_with_no_diagonal_falls_back_to_a_nonzero_scale():
    """An orphan DOF -- one no volume term touched -- has |A_ii| = 0. Scaling by that would leave a
    zero row and a singular matrix, so the fallback must be a nonzero matrix-scale value."""
    K = _laplacian()
    K[2, :] = 0.0
    K[:, 2] = 0.0  # DOF 2 is now completely disconnected
    D, b = _eliminated(K, pins=[(2, 0.75)], rhs=np.ones(K.shape[0]))
    assert D[2, 2] > 0.0, "an orphan Dirichlet DOF was scaled to zero -- the matrix is singular"
    assert np.isclose(np.linalg.solve(D, b)[2], 0.75, rtol=0, atol=1e-12)
