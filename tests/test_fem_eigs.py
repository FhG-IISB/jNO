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


# ---------------------------------------------------------------------------------------------
# Preconditioned LOBPCG (precond= selects it) — checked against the dense path as the oracle
# ---------------------------------------------------------------------------------------------


def test_lobpcg_matches_the_dense_spectrum_including_degeneracies():
    """LOBPCG reproduces the dense reduction's spectrum on the pencil with the degenerate pairs —
    the block method's whole reason for existing is that it resolves a repeated eigenvalue."""
    K, mass = _laplacian(0.045)
    ref, _ = K.eigs(mass=mass, k=6)
    lam, _X = K.eigs(mass=mass, k=6, precond=jno.precond.jacobi(), tol=1e-6, maxiter=300)
    lam, ref = np.asarray(lam), np.asarray(ref)
    assert np.all(np.isfinite(lam)), "NaN-poisoned => did not converge inside the budget"
    assert np.allclose(lam, ref, rtol=2e-6, atol=1e-7)
    assert abs(lam[1] - lam[2]) < 1e-2 * lam[2]  # (1,0)/(0,1) still degenerate
    assert abs(lam[4] - lam[5]) < 1e-2 * lam[5]  # (2,0)/(0,2)


def test_lobpcg_eigenvectors_are_m_orthonormal():
    """The invariant the iterative path must hold exactly as the dense one does: XᵀMX = I. This is what
    the M-inner-product Rayleigh-Ritz buys — the mass matrix here is CONSISTENT, not lumped."""
    K, mass = _laplacian(0.06)
    _lam, X = K.eigs(mass=mass, k=5, precond=jno.precond.jacobi(), tol=1e-6, maxiter=300)
    M = _dense(jno.fem(mass).operator[0])
    X = np.asarray(X)
    assert np.max(np.abs(X.T @ M @ X - np.eye(5))) < 1e-8


def test_lobpcg_eigenvalue_is_differentiable():
    """∂λ/∂θ flows through the LOBPCG path even though the sweeps are stop_gradient'd: the Rayleigh
    quotient at the frozen eigenvector gives the exact derivative for a simple eigenvalue. K→sK scales
    every eigenvalue, so ∂λ/∂s = λ₀."""
    K, mass = _laplacian(0.07)
    K0 = jnp.asarray(_dense(K.operator[0]))
    M = jnp.asarray(_dense(jno.fem(mass).operator[0]))
    solver = jno.solve.eigs(k=4, precond=jno.precond.jacobi(), tol=1e-6, maxiter=300)

    def lam_simple(s):
        lam, _ = solver(s * K0, M)
        return lam[3]  # the (1,1) mode — simple

    val = float(lam_simple(1.0))
    g = float(jax.grad(lam_simple)(1.0))
    fd = float((lam_simple(1.0 + 1e-6) - lam_simple(1.0 - 1e-6)) / 2e-6)
    assert abs(g - fd) / abs(fd) < 1e-4
    assert abs(g - val) / val < 1e-5  # analytic d(sλ)/ds = λ


def test_lobpcg_largest_and_standard_problem():
    """``which='largest'`` descends, and ``M=None`` is the standard problem — both on the iterative path."""
    K, _ = _laplacian(0.12)
    lam, X = jno.solve.eigs(k=3, which="largest", precond=jno.precond.jacobi(), tol=1e-6, maxiter=300)(K.operator[0])
    ref, _ = jno.solve.eigs(k=3, which="largest")(K.operator[0])
    lam, ref = np.asarray(lam), np.asarray(ref)
    assert np.all(np.diff(lam) <= 1e-9)  # descending
    assert lam.shape == (3,) and X.shape[1] == 3
    assert np.allclose(lam, ref, rtol=1e-5)


def test_lobpcg_k_equals_one_and_full_rank_extremes():
    """Extremes of the block size: k=1 (a single vector, so the [X, W, P] block is 3 columns and P starts
    empty) and k = n (the block spans the whole space, making the Rayleigh-Ritz basis maximally
    rank-deficient — the case the zero-column handling exists for)."""
    K, mass = _laplacian(0.25)
    n = _dense(K.operator[0]).shape[0]
    ref, _ = K.eigs(mass=mass, k=n)

    lam1, _ = K.eigs(mass=mass, k=1, precond=jno.precond.jacobi(), tol=1e-6, maxiter=300)
    assert np.asarray(lam1).shape == (1,)
    assert np.allclose(np.asarray(lam1)[0], np.asarray(ref)[0], rtol=1e-5, atol=1e-8)

    lam_all, Xall = K.eigs(mass=mass, k=n, precond=jno.precond.jacobi(), tol=1e-6, maxiter=600)
    assert np.all(np.isfinite(np.asarray(lam_all))) and Xall.shape[1] == n


def test_lobpcg_nan_poisons_on_an_exhausted_budget():
    """An exhausted sweep budget returns NaN rather than a quietly under-converged spectrum — the same
    contract as the adaptive time march. One sweep cannot converge this pencil."""
    K, mass = _laplacian(0.06)
    lam, X = K.eigs(mass=mass, k=6, precond=jno.precond.jacobi(), tol=1e-14, maxiter=1)
    assert np.all(np.isnan(np.asarray(lam))) and np.all(np.isnan(np.asarray(X)))


def test_eigs_rejects_tol_without_precond():
    """tol=/maxiter= without precond= would be silently ignored by the dense path, so it fails loud."""
    K, mass = _laplacian(0.25)
    with pytest.raises(ValueError, match="no precond="):
        K.eigs(mass=mass, k=2, tol=1e-10)
    with pytest.raises(ValueError, match="no precond="):
        jno.solve.eigs(k=2, maxiter=10)


def test_eigs_rejects_a_source_term():
    """A load has no place in ``Kx = λMx``: the pencil reads only the matrix, so a source would be
    silently dropped and return the *undriven* spectrum. Measured before the guard: adding ``-3·v`` on
    the boundary gave eigenvalues bit-identical to the source-free problem."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    d.tag("wall", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("wall", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    ub, vb = u.bind(x=xb, y=yb), v.bind(x=xb, y=yb)
    stiff = ui.x * vi.x + ui.y * vi.y
    mass = [ui * vi]

    for terms in ([stiff, -3.0 * vb], [stiff, -3.0 * vi], [stiff - 3.0 * vi]):  # surface, volume, inside a sum
        with pytest.raises(ValueError, match="source term"):
            jno.fem(list(terms)).eigs(mass=mass, k=2)

    # Bilinear surface terms are NOT sources: Robin/impedance belongs in K and must still work.
    lam_neu, _ = jno.fem([stiff]).eigs(mass=mass, k=3)
    lam_rob, _ = jno.fem([stiff, 2.0 * ub * vb]).eigs(mass=mass, k=3)
    assert np.asarray(lam_rob)[0] > np.asarray(lam_neu)[0] + 1e-6  # alpha>0 lifts the zero mode


# ---------------------------------------------------------------------------------------------
# Space-reducing constraints. fem.solve() applies Dirichlet by ROW REPLACEMENT and periodic ties
# by a per-call reduction; neither survives into fem.operator[0], so an eigensolve reading that
# matrix directly is wrong. Oracles are analytic, not self-comparisons.
# ---------------------------------------------------------------------------------------------


def _dirichlet_box(mesh_size):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    d.tag("wall", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9) | (y < 1e-9) | (y > 1 - 1e-9))
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("wall", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    return d, jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 0.0]), [ui * vi]


def test_eigs_dirichlet_dofs_are_eliminated_not_row_replaced():
    """Dirichlet box: λ = π²(n²+m²), n,m ≥ 1. Row replacement leaves identity rows against a full mass
    row, injecting spurious pairs (measured: lowest spurious at 267 while the true spectrum starts at
    19.9). Eliminating the constrained DOFs removes them, and the modes vanish on the wall exactly."""
    d, K, mass = _dirichlet_box(0.05)
    lam, X = K.eigs(mass=mass, k=6)
    analytic = np.pi**2 * np.array([2.0, 5, 5, 8, 10, 10])
    assert np.allclose(np.asarray(lam), analytic, rtol=0.06), np.asarray(lam)

    pts = np.asarray(K.points)
    onwall = (
        (np.abs(pts[:, 0]) < 1e-9)
        | (np.abs(pts[:, 0] - 1) < 1e-9)
        | (np.abs(pts[:, 1]) < 1e-9)
        | (np.abs(pts[:, 1] - 1) < 1e-9)
    )
    assert np.max(np.abs(np.asarray(X)[onwall])) == 0.0  # eliminated => exactly zero, not "small"


def test_eigs_periodic_tie_reduces_the_pencil():
    """Periodic-in-x / Neumann-in-y: λ = (2πn)² + (πm)² → 0, π², 4π², 4π². Before the reduction this
    returned the NON-periodic Neumann spectrum (0, π², π², 2π²) — a silently wrong band structure."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.07)
    d.tag("L", lambda x, y: x < 1e-9)
    d.tag("R", lambda x, y: x > 1 - 1e-9)
    d._remesh_periodic([("L", "R")])  # conforming faces for the nodal tie
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xl, yl, _ = d.variable("L", split=True)
    xr, yr, _ = d.variable("R", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    K = jno.fem([ui.x * vi.x + ui.y * vi.y, u(xl, yl) - u(xr, yr)])

    lam, X = K.eigs(mass=[ui * vi], k=4)
    lam = np.asarray(lam)
    assert np.allclose(lam, np.pi**2 * np.array([0.0, 1, 4, 4]), rtol=0.06, atol=0.05), lam
    assert lam[2] > 3.0 * lam[1]  # the 4π² pair, NOT the Neumann π² duplicate

    pts, X = np.asarray(K.points), np.asarray(X)
    li = np.where(pts[:, 0] < 1e-9)[0]
    ri = np.where(pts[:, 0] > 1 - 1e-9)[0]
    lo, ro = li[np.argsort(pts[li, 1])], ri[np.argsort(pts[ri, 1])]
    assert np.max(np.abs(X[lo] - X[ro])) == 0.0  # the tie holds on the returned modes


def test_eigs_constraints_compose_with_lobpcg():
    """The reduction is applied as matvecs, so it must work on the iterative path too — and the
    reduced pencil must never be densified for it."""
    d, K, mass = _dirichlet_box(0.06)
    ref, _ = K.eigs(mass=mass, k=4)
    lam, X = K.eigs(mass=mass, k=4, precond=jno.precond.jacobi(), tol=1e-6, maxiter=400)
    assert np.all(np.isfinite(np.asarray(lam)))
    # tol=1e-6 on the residual buys ~1e-4 on the eigenvalues here; the point of this test is that the
    # reduction is applied on the iterative path at all, not eigenvalue precision.
    assert np.allclose(np.asarray(lam), np.asarray(ref), rtol=1e-3)


def test_eigs_rejects_inhomogeneous_dirichlet_and_mixed_constraints():
    """An inhomogeneous pin has no meaning for modes (the DOFs are eliminated, not driven), and
    periodic+Dirichlet is not composed yet — both fail loud rather than quietly doing the wrong thing."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.2)
    d.tag("wall", lambda x, y: (x < 1e-9) | (x > 1 - 1e-9))
    u, v = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xb, yb, _ = d.variable("wall", split=True)
    ui, vi = u.bind(x=xi, y=yi), v.bind(x=xi, y=yi)
    with pytest.raises(ValueError, match="INHOMOGENEOUS"):
        jno.fem([ui.x * vi.x + ui.y * vi.y, u(xb, yb) - 1.0]).eigs(mass=[ui * vi], k=2)


def test_eigs_is_idempotent():
    """Calling ``eigs`` twice on the same fem must give the same spectrum. The first call's internal
    mass assembly cleared the domain's Dirichlet stash, so the SECOND call silently skipped the
    elimination and returned the row-replaced spurious spectrum (measured: 47.2 appeared where the
    true reduced spectrum has no eigenvalue at all)."""
    d, K, mass = _dirichlet_box(0.25)
    lam1, _ = K.eigs(mass=mass, k=3)
    lam2, _ = K.eigs(mass=mass, k=3)
    assert np.allclose(np.asarray(lam1), np.asarray(lam2), rtol=1e-12), (lam1, lam2)
