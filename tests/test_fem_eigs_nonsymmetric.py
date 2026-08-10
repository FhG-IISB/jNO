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


def test_differentiating_raises_rather_than_returning_a_wrong_number():
    """`pure_callback` has no JVP rule. The requirement is that this FAILS, not that it silently
    produces zeros -- the symmetric paths are the differentiable ones."""
    A = _nonsym(100, seed=17)

    def loss(a):
        return jnp.sum(jnp.abs(nonsymmetric_geneigh(a, None, 2, 5.0)[0]))

    with pytest.raises(Exception):
        jax.grad(loss)(jnp.asarray(A))
