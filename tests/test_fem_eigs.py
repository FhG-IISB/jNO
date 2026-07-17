"""``jno.solve.eigs`` / ``FEM.eigs`` — generalized symmetric eigensolver ``K x = λ M x`` (V1a: dense).

Oracle is the all-Neumann box Laplacian, whose spectrum ``λ = π²(n₁²+n₂²)`` is analytic and *degenerate*
((1,0)/(0,1), (2,0)/(0,2)) — so it stresses the block/degeneracy handling. Guards: the spectrum, the
M-orthonormality invariant ``XᵀMX=I``, and a differentiable (simple) eigenvalue vs finite differences.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from shapely.geometry import box

import jno


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _laplacian(mesh_size):
    """Stiffness fem K = ∫∇u·∇v and the mass form [u v] on the unit square (all-Neumann)."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    return jno.fem([ui.x * vi.x + ui.y * vi.y]), [ui * vi]


def _dense(A):
    return np.asarray(A.todense() if hasattr(A, "todense") else A)


def test_eigs_box_laplacian_spectrum_with_degeneracies():
    """The six lowest generalized eigenvalues match π²[0,1,1,2,4,4] and the degenerate pairs coincide."""
    K, mass = _laplacian(0.045)
    lam, X = K.eigs(mass=mass, k=6)
    lam = np.asarray(lam)
    analytic = np.pi**2 * np.array([0.0, 1, 1, 2, 4, 4])
    assert np.allclose(lam, analytic, rtol=0.06, atol=0.3)  # P1 over-estimates slightly; loose gate
    assert abs(lam[1] - lam[2]) < 1e-2 * lam[2]  # (1,0)/(0,1) degeneracy
    assert abs(lam[4] - lam[5]) < 1e-2 * lam[5]  # (2,0)/(0,2) degeneracy


def test_eigs_eigenvectors_are_m_orthonormal():
    """Eigenvectors are M-orthonormal: XᵀMX = I (the invariant every eigensolver variant must hold)."""
    K, mass = _laplacian(0.06)
    lam, X = K.eigs(mass=mass, k=6)
    M = _dense(jno.fem(mass).operator[0])
    assert np.max(np.abs(np.asarray(X).T @ M @ np.asarray(X) - np.eye(6))) < 1e-9


def test_eigs_eigenvalue_is_differentiable():
    """∂λ/∂θ flows through the solve. Scaling K→sK scales every eigenvalue (λ = s·λ₀), so for a simple
    eigenvalue ∂λ/∂s = λ₀; the autodiff gradient matches both finite differences and that analytic value."""
    K, mass = _laplacian(0.07)
    K0 = jnp.asarray(_dense(K.operator[0]))
    M = jnp.asarray(_dense(jno.fem(mass).operator[0]))
    solver = jno.solve.eigs(k=6)

    def lam_simple(s):
        lam, _ = solver(s * K0, M)
        return lam[3]  # the (1,1) mode ~2π² — simple (non-degenerate)

    val = float(lam_simple(1.0))
    g = float(jax.grad(lam_simple)(1.0))
    fd = float((lam_simple(1.0 + 1e-6) - lam_simple(1.0 - 1e-6)) / 2e-6)
    assert abs(g - fd) / abs(fd) < 1e-4  # matches finite differences
    assert abs(g - val) / val < 1e-5  # and the analytic d(sλ)/ds = λ


def test_eigs_standard_problem_and_largest():
    """``M=None`` is the standard problem ``Kx=λx``; ``which='largest'`` returns them in descending order."""
    K, _ = _laplacian(0.12)
    lam, X = jno.solve.eigs(k=3, which="largest")(K.operator[0])
    lam = np.asarray(lam)
    assert np.all(np.diff(lam) <= 1e-9)  # descending
    assert lam.shape == (3,) and X.shape[1] == 3
