"""``jno.solve.eigs`` on a NON-self-adjoint pencil: complex spectrum via ARPACK/Arnoldi.

The point of these tests is the thing the symmetric solver got *wrong* by construction. Both symmetric
reductions Hermitianize, so on a non-symmetric ``K`` they return the spectrum of ``½(K+Kᵀ)`` -- a
different problem, none of whose values need be an eigenvalue of ``K`` at all. So the oracle here is
always a dense reference on the ORIGINAL matrix, and several tests assert the answer is genuinely
complex, which the symmetrized surrogate can never be.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jno
from jno.utils.solver.eigen import _symmetry_verdict, nonsymmetric_geneigh


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _nonsym(n, seed=0):
    """Non-normal, with genuinely complex eigenvalues (a convection-like upper shift)."""
    rng = np.random.default_rng(seed)
    A = np.diag(np.linspace(1.0, 20.0, n)) + 2.0 * np.diag(np.ones(n - 1), 1)
    A += rng.normal(0, 0.05, (n, n)) * (rng.random((n, n)) < 0.02)
    return A


def _dense_nearest(A, sigma, k):
    lam = np.linalg.eigvals(A)
    return np.sort_complex(lam[np.argsort(np.abs(lam - sigma))[:k]])


# ---------------------------------------------------------------------------------
# the verdict that does the routing
# ---------------------------------------------------------------------------------


def test_verdict_distinguishes_symmetric_from_not():
    assert _symmetry_verdict(jnp.asarray([[2.0, 1.0], [1.0, 3.0]])) == "symmetric"
    assert _symmetry_verdict(jnp.asarray([[2.0, 1.0], [5.0, 3.0]])) == "nonsymmetric"
    assert _symmetry_verdict(None) == "symmetric"  # an absent mass matrix is not a problem


def test_verdict_declines_to_guess_under_a_trace():
    """A traced operator cannot be concretised; fabricating a verdict would route on a lie."""
    seen = {}

    def f(x):
        seen["v"] = _symmetry_verdict(x)
        return jnp.sum(x)

    jax.jit(f)(jnp.asarray([[2.0, 1.0], [5.0, 3.0]]))
    assert seen["v"] == "unknown"


# ---------------------------------------------------------------------------------
# the spectrum itself
# ---------------------------------------------------------------------------------


def test_interior_eigenvalues_match_a_dense_reference_and_are_complex():
    n = 200
    A = _nonsym(n)
    lam, V = jno.solve.eigs(k=4, sigma=10.0)(jnp.asarray(A))
    got = np.sort_complex(np.asarray(lam))
    np.testing.assert_allclose(got, _dense_nearest(A, 10.0, 4), rtol=1e-8, atol=1e-9)
    assert np.any(np.abs(got.imag) > 1e-6), "spectrum came back real; the surrogate would too"
    resid = np.max(np.abs(A @ np.asarray(V) - np.asarray(V) * np.asarray(lam)[None, :]))
    assert resid < 1e-10


def test_the_answer_is_NOT_the_symmetrized_surrogate():
    """The specific wrongness this path exists to avoid."""
    n = 120
    A = _nonsym(n, seed=3)
    lam = np.asarray(jno.solve.eigs(k=3, sigma=8.0)(jnp.asarray(A))[0])
    sym = np.linalg.eigvalsh(0.5 * (A + A.T))
    for value in lam:
        assert np.min(np.abs(sym - value)) > 1e-8, "returned a value from ½(K+Kᵀ)"


def test_dtype_is_complex_even_when_the_spectrum_happens_to_be_real():
    """A real return would misrepresent what a non-self-adjoint operator can produce."""
    A = np.diag([1.0, 2.0, 3.0, 4.0, 5.0]) + np.diag([0.1, 0.1, 0.1, 0.1], 1)  # non-symmetric, real spectrum
    lam, _ = jno.solve.eigs(k=2, sigma=2.5)(jnp.asarray(A))
    assert jnp.iscomplexobj(lam)


def test_growth_rate_ordering_is_available():
    """`LR` is what a stability question actually asks: the rightmost eigenvalue in the complex plane."""
    n = 100
    A = _nonsym(n, seed=5)
    lam = np.asarray(nonsymmetric_geneigh(jnp.asarray(A), None, 3, None, "LR")[0])
    ref = np.linalg.eigvals(A)
    np.testing.assert_allclose(np.sort(lam.real)[::-1], np.sort(ref.real)[::-1][:3], rtol=1e-8)


def test_generalized_pencil_against_a_dense_reference():
    import scipy.linalg as sla

    n = 90
    A = _nonsym(n, seed=7)
    M = np.diag(np.linspace(1.0, 2.0, n))
    lam = np.asarray(nonsymmetric_geneigh(jnp.asarray(A), jnp.asarray(M), 3, 9.0)[0])
    ref = sla.eig(A, M)[0]
    ref = np.sort_complex(ref[np.argsort(np.abs(ref - 9.0))[:3]])
    np.testing.assert_allclose(np.sort_complex(lam), ref, rtol=1e-7, atol=1e-8)


def test_small_pencil_takes_the_exact_dense_path():
    A = _nonsym(12, seed=9)
    lam = np.asarray(jno.solve.eigs(k=3, sigma=5.0)(jnp.asarray(A))[0])
    np.testing.assert_allclose(np.sort_complex(lam), _dense_nearest(A, 5.0, 3), rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------------
# no regression on the symmetric side, and loud limits on the new one
# ---------------------------------------------------------------------------------


def test_symmetric_operators_still_take_the_symmetric_path():
    n = 150
    rng = np.random.default_rng(11)
    Q = np.linalg.qr(rng.normal(size=(n, n)))[0]
    K = Q @ np.diag(np.linspace(1.0, 40.0, n)) @ Q.T
    K = 0.5 * (K + K.T)
    # BCOO, not dense: shift-invert forms K - sigma*I sparsely and a dense operand needs a concrete
    # nse it cannot get under the trace. fem.eigs always hands over assembled sparse operators.
    from jax.experimental import sparse as jsp

    lam, _ = jno.solve.eigs(k=3, sigma=12.0)(jsp.BCOO.fromdense(jnp.asarray(K)))
    assert not jnp.iscomplexobj(lam), "a symmetric pencil must keep the real, differentiable path"
    ref = np.linalg.eigvalsh(K)
    np.testing.assert_allclose(np.sort(np.asarray(lam)), np.sort(ref[np.argsort(np.abs(ref - 12.0))[:3]]), rtol=1e-6)


def test_linear_slot_is_refused_not_ignored():
    """Silently dropping a solver the caller asked for is the failure mode jNO refuses."""
    A = _nonsym(100, seed=13)
    with pytest.raises(ValueError, match="non-symmetric"):
        jno.solve.eigs(k=3, sigma=5.0, linear=jno.solve.lu())(jnp.asarray(A))


def test_precond_slot_is_refused_not_ignored():
    """Without sigma there is no pre-existing sigma/precond guard, so this reaches the Arnoldi check."""
    A = _nonsym(100, seed=13)
    with pytest.raises(ValueError, match="non-symmetric"):
        jno.solve.eigs(k=3, precond=jno.precond.jacobi())(jnp.asarray(A))


def test_unknown_which_names_the_options_including_the_growth_rate():
    A = _nonsym(100, seed=15)
    with pytest.raises(ValueError, match="LR"):
        nonsymmetric_geneigh(jnp.asarray(A), None, 3, None, "algebraically_smallest")


# ---------------------------------------------------------------------------------
# linear= drives ARPACK's shift-invert factorization
# ---------------------------------------------------------------------------------
#
# ARPACK asks for (K - sigma*M)^-1 as an operator and applies it ~50-70 times per run, so this is the
# "factor once, solve many" shape. It works at all only because the backends' host kernels are plain
# numpy functions: ARPACK calls back into Python from Fortran, outside any JAX trace.


def test_host_backend_gives_the_same_spectrum_as_the_default():
    n = 300
    A = _nonsym(n, seed=21)
    ref = np.sort_complex(np.asarray(jno.solve.eigs(k=4, sigma=10.0)(jnp.asarray(A))[0]))
    got = np.sort_complex(
        np.asarray(jno.solve.eigs(k=4, sigma=10.0, linear=jno.solve.lu(backend="host"))(jnp.asarray(A))[0])
    )
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-10)


def test_device_backend_is_refused_because_arpack_calls_from_host_code():
    A = _nonsym(200, seed=23)
    with pytest.raises(ValueError, match="JAX primitive"):
        jno.solve.eigs(k=3, sigma=5.0, linear=jno.solve.lu(backend="device"))(jnp.asarray(A))


def test_an_iterative_solver_is_refused_with_the_reason():
    A = _nonsym(200, seed=25)
    with pytest.raises(ValueError, match="iterative solver"):
        jno.solve.eigs(k=3, sigma=5.0, linear=jno.solve.bicgstab())(jnp.asarray(A))


def test_linear_without_sigma_is_refused():
    """No shift means no (K - sigma*M) to factor -- Arnoldi runs on plain matvecs."""
    A = _nonsym(200, seed=27)
    with pytest.raises(ValueError, match="shift-invert"):
        jno.solve.eigs(k=3, linear=jno.solve.lu(backend="host"))(jnp.asarray(A))


@pytest.mark.parametrize("backend", ["pardiso", "cudss"])
def test_accelerated_backends_agree_with_the_default_when_installed(backend):
    from jno.utils.solver.linear import _cudss_available, _pardiso_available

    if not {"pardiso": _pardiso_available, "cudss": _cudss_available}[backend]():
        pytest.skip(f"optional backend {backend} not installed")
    n = 400
    A = _nonsym(n, seed=29)
    ref = np.sort_complex(np.asarray(jno.solve.eigs(k=4, sigma=10.0)(jnp.asarray(A))[0]))
    got = np.sort_complex(
        np.asarray(jno.solve.eigs(k=4, sigma=10.0, linear=jno.solve.lu(backend=backend))(jnp.asarray(A))[0])
    )
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-9)


# ---------------------------------------------------------------------------------
# eigenvalue derivatives
# ---------------------------------------------------------------------------------
#
# dλ = wᴴ(dA − λ dB)v / (wᴴBv) for a SIMPLE eigenvalue (Wilkinson 1965, ch. 2), which needs the LEFT
# eigenvector w. Every assertion below is against finite differences on the ORIGINAL matrix, and each
# guard test names the wrong answer it is there to prevent -- all three were observed during
# development, so none of them is hypothetical.


def _fd(f, A, i, j, eps=1e-5):
    Ap, Am = A.copy(), A.copy()
    Ap[i, j] += eps
    Am[i, j] -= eps
    return (f(jnp.asarray(Ap)) - f(jnp.asarray(Am))) / (2 * eps)


@pytest.mark.parametrize("n,label", [(40, "dense path"), (200, "sparse path (inverse iteration)")])
def test_eigenvalue_gradient_matches_finite_differences(n, label):
    A = _nonsym(n, seed=31)

    def f(a):
        return jnp.sum(jnp.real(nonsymmetric_geneigh(a, None, 3, 5.0)[0]))

    g = np.asarray(jax.grad(f)(jnp.asarray(A)))
    assert not np.isnan(g).any()
    for i, j in [(0, 0), (1, 2), (3, 1)]:
        # FD is the imprecise side here: central differences carry ~1e-8 absolute error, which on a
        # derivative of order 1e-05 is already 1e-03 relative. The analytic value is the better one.
        np.testing.assert_allclose(g[i, j], _fd(f, A, i, j), rtol=5e-3, atol=1e-8)


def test_eigenvalue_gradient_for_a_generalized_pencil():
    n = 150
    A = _nonsym(n, seed=33)
    M = jnp.asarray(np.diag(np.linspace(1.0, 2.0, n)))

    def f(a):
        return jnp.sum(jnp.real(nonsymmetric_geneigh(a, M, 3, 5.0)[0]))

    g = np.asarray(jax.grad(f)(jnp.asarray(A)))
    for i, j in [(0, 0), (2, 4)]:
        np.testing.assert_allclose(g[i, j], _fd(f, A, i, j), rtol=5e-3, atol=1e-8)


def test_a_defective_eigenvalue_gives_NaN_not_a_huge_finite_number():
    """A Jordan block's eigenvalue has NO derivative -- its perturbation series runs in sqrt(eps).

    Two wrong answers were produced here before this worked. With the condition-number threshold set
    to 1e-12 the guard never fired and the formula returned **1.6e+08**. With the NaN in a
    `where` BRANCH it returned exactly **0.0**, because a constant branch transposes to zero under
    reverse mode. The NaN has to divide, so that it stays linear in the tangent and survives.
    """
    J = np.kron(np.eye(30), np.array([[2.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 5.0]]))

    def f(a):
        return jnp.sum(jnp.real(nonsymmetric_geneigh(a, None, 2, 2.1)[0]))

    g = np.asarray(jax.grad(f)(jnp.asarray(J)))
    assert np.isnan(g).any(), "a defective eigenvalue must not yield a finite gradient"


def test_eigenvector_gradients_are_NaN_not_a_silent_zero():
    """The eigenvectors come from a stop_gradient-ed callback, so the DEFAULT behaviour would be a
    silent zero -- measured, the true derivative was 2.4e-04. NaN is the loud alternative."""
    n = 80
    A = _nonsym(n, seed=35)
    g = np.asarray(jax.grad(lambda a: jnp.sum(jnp.abs(nonsymmetric_geneigh(a, None, 2, 5.0)[1])))(jnp.asarray(A)))
    assert np.isnan(g).all()
    assert not np.all(g == 0)


def test_an_eigenvalue_only_loss_is_unaffected_by_the_eigenvector_guard():
    """The guard must key on whether the eigenvectors are ACTUALLY used. A custom_jvp version fired
    during tracing -- before dead-code elimination -- and broke this case."""
    n = 80
    A = _nonsym(n, seed=37)

    def f(a):
        return jnp.sum(jnp.real(nonsymmetric_geneigh(a, None, 2, 5.0)[0]))

    g = np.asarray(jax.grad(f)(jnp.asarray(A)))
    assert not np.isnan(g).any()
    np.testing.assert_allclose(g[1, 2], _fd(f, A, 1, 2), rtol=5e-3, atol=1e-8)


def test_forward_mode_is_refused_rather_than_silently_wrong():
    """The eigenvector guard is a custom_vjp, so jax.jvp/jacfwd are unavailable here. Documented,
    and reverse mode is what an inverse problem uses."""
    A = _nonsym(60, seed=39)
    with pytest.raises(TypeError, match="custom_vjp"):
        jax.jvp(lambda a: nonsymmetric_geneigh(a, None, 2, 5.0)[0], (jnp.asarray(A),), (jnp.ones_like(jnp.asarray(A)),))
