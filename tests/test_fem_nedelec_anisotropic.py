"""Anisotropic permittivity on a Nédélec (N1E) Maxwell field — a 3×3 tensor ε̂ instead of a scalar.

The mass term is authored ``inner(ε̂ @ u, v)`` (a ``MatrixView`` applied to the vector trial), giving the
weak form ``∫ (ε̂·E)·v``. Because ``M @ u`` is linear in the unknown (the matrix is a coefficient), the
form assembles as a steady linear system — enabling birefringence / waveplates / liquid-crystal / gyrotropic
media in ``jno.fem``. Any tensor works: diagonal (uniaxial/biaxial), full symmetric, or complex non-symmetric.
"""

import numpy as np
import pytest

pytest.importorskip("pygmsh", reason="pygmsh required for 3D cube meshing")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import jno  # noqa: E402
from jno.trace.views import MatrixView  # noqa: E402

inner, vec = jno.np.inner, jno.np.vector
_dense = lambda A: np.asarray(jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A))  # noqa: E731
_arr = lambda x: np.asarray(jnp.asarray(x)).reshape(-1)  # noqa: E731


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _cube(mesh_size=0.5):
    d = jno.domain(constructor=jno.domain.cube(mesh_size=mesh_size))
    u, v = d.fem_symbols(value_shape=(3,), names=("u", "v"), space="N1E")
    c = d.variable("interior", split=True)
    xi, yi, zi = c[0], c[1], c[2]
    ui, vi = u.bind(x=xi, y=yi, z=zi), v.bind(x=xi, y=yi, z=zi)
    cu, cv = u.vector.curl(xi, yi, zi), v.vector.curl(xi, yi, zi)
    return d, (xi, yi, zi), (ui, vi), (cu, cv)


def test_uniaxial_quadratic_form_picks_the_axis_component():
    """Decisive physics check. For a uniaxial ε̂ = diag(εₒ, εₒ, εₑ), the mass ∫(ε̂·E)·E on the unit cube with
    a CONSTANT field (exact in N1E0) equals εₑ for E=(0,0,1) and εₒ for E=(1,0,0) — the tensor applies the
    right principal value per direction. Uses uᵀ·M_aniso·u on the projected constant field."""
    eo, ee = 2.0, 5.0
    d, (xi, yi, zi), (ui, vi), _ = _cube(0.4)
    eps = MatrixView(vec(eo + 0 * xi, eo + 0 * yi, ee + 0 * zi).expr).from_diag()
    M_iso = _dense(jno.fem([inner(ui, vi)]).A)  # standard mass, to project constants
    M_an = _dense(jno.fem([inner(eps @ ui, vi)]).A)  # anisotropic mass ∫(ε̂ φ_a)·φ_b

    for field, expect in [(vec(0.0 * xi, 0.0 * yi, 1.0 + 0.0 * zi), ee), (vec(1.0 + 0.0 * xi, 0.0 * yi, 0.0 * zi), eo)]:
        u_dof = np.linalg.solve(M_iso, _arr(jno.fem([inner(ui, vi) - inner(field, vi)]).b))  # exact projection
        np.testing.assert_allclose(float(u_dof @ M_an @ u_dof), expect, atol=1e-9)


def test_full_symmetric_tensor_assembles_symmetric_and_is_genuinely_anisotropic():
    """A full symmetric, spatially-varying ε̂ (off-diagonal, x/y/z-dependent) assembles to a symmetric matrix
    that differs from BOTH isotropic limits — genuine anisotropy, not a disguised scalar."""
    d, (xi, yi, zi), (ui, vi), (cu, cv) = _cube(0.5)
    e00, e11, e22 = 3.0 + xi, 4.0 + yi, 5.0 - zi
    e01, e02, e12 = 0.5 * yi, 0.3 + 0.0 * xi, 0.2 * zi
    eps = MatrixView(vec(e00, e01, e02, e01, e11, e12, e02, e12, e22).expr).from_flat(3, 3)

    A = _dense(jno.fem([inner(cu, cv) - inner(eps @ ui, vi)]).A)
    np.testing.assert_allclose(A, A.T, atol=1e-7)  # symmetric ε̂ → symmetric operator
    A_lo = _dense(jno.fem([inner(cu, cv) - 3.0 * inner(ui, vi)]).A)
    A_hi = _dense(jno.fem([inner(cu, cv) - 5.0 * inner(ui, vi)]).A)
    assert not np.allclose(A, A_lo, atol=1e-6) and not np.allclose(A, A_hi, atol=1e-6)


def test_gyrotropic_tensor_is_complex_and_not_misclassified_nonlinear():
    """A gyrotropic ε̂ (Hermitian, imaginary off-diagonal ε_xy = -ε_yx = ig) assembles as a COMPLEX steady
    LINEAR system — ε̂ @ u is linear in u, so it must not be misread as a nonlinear form."""
    g = 0.8
    d, (xi, yi, zi), (ui, vi), (cu, cv) = _cube(0.6)
    o, z = 1.0 + 0.0 * xi, 0.0 * xi
    eps = MatrixView(vec(3 * o, 1j * g * o, z, -1j * g * o, 3 * o, z, z, z, 3 * o).expr).from_flat(3, 3)

    fem = jno.fem([inner(cu, cv) - inner(eps @ ui, vi)])
    assert fem.is_complex  # complex tensor → complex system (steady linear, not nonlinear)
    op_r, op_i = fem._op
    assert np.max(np.abs(_dense(op_i[0]))) > 1e-6  # the imaginary (gyrotropic) part survives assembly
