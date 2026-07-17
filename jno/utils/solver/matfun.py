"""Matrix functions and spectral quantities — **log-determinant, trace, and ``f(A)·v``** — for large
operators, via the optional **matfree** package (N. Krämer, MIT; https://pnkraemer.github.io/matfree/).

Everything here is **matrix-free** (it touches the operator only through its matvec, so it scales to
problems too large to factor) and **differentiable** (matfree carries the JVP/VJP through its Lanczos /
Arnoldi iterations — no hand-written adjoint). ``logdet`` / ``trace`` are *stochastic* estimators
(Hutchinson probes + stochastic Lanczos quadrature): the return value is an unbiased estimate whose
variance falls with ``samples`` (probe vectors) and whose bias falls with ``order`` (Lanczos steps).

These unlock, differentiably, things ``Ax=b`` solvers cannot express: Bayesian **log-evidence /
marginal likelihood** (``logdet`` of a FEM precision), **uncertainty / effective-DOF** diagnostics
(``trace``), and **exponential time integrators** (``exp(-dt·A)·u`` via ``applyfun``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _require_matfree():
    try:
        import matfree  # noqa: F401
    except ImportError as e:  # optional dependency — keep core jNO lean
        raise ImportError(
            "jno.solve.logdet / trace / applyfun / diagonal need the optional 'matfree' package "
            "(MIT, pure JAX). Install it with:  pip install matfree"
        ) from e


def _operator(A):
    """(matvec, n, dtype) for a jNO ``LinearOperator`` / BCOO / dense matrix — the matrix-free view."""
    from .solver_api import LinearOperator

    mv = A.mv if isinstance(A, LinearOperator) or hasattr(A, "mv") else (lambda v: A @ v)
    n = A.shape[0]
    dtype = jax.eval_shape(mv, jnp.zeros(n)).dtype  # matches the operator's field (real / complex)
    return mv, n, dtype


def _key(key):
    return jax.random.PRNGKey(0) if key is None else key


def logdet(A, *, samples: int = 32, order: int = 25, key=None):
    """Differentiable stochastic estimate of ``log det A`` for a symmetric positive-definite ``A``
    (stochastic Lanczos quadrature). ``samples`` probe vectors, ``order`` Lanczos steps."""
    _require_matfree()
    from matfree import decomp, funm, stochtrace

    mv, n, dtype = _operator(A)
    integrand = funm.monte_carlo_funm_sym_logdet(decomp.tridiag_sym(order))
    estimate = stochtrace.estimator_monte_carlo(
        integrand, sampler=stochtrace.sampler_normal(jnp.zeros(n, dtype), num=samples)
    )
    return estimate(mv, _key(key))


def trace(A, *, fun=None, samples: int = 32, order: int = 25, key=None):
    """Differentiable stochastic estimate of ``tr A`` (Hutchinson), or ``tr f(A)`` when ``fun`` is a
    scalar function (via stochastic Lanczos quadrature; ``A`` symmetric). ``fun=jnp.log`` reproduces
    :func:`logdet`; ``fun=lambda z: 1/z`` gives ``tr(A⁻¹)``."""
    _require_matfree()
    from matfree import decomp, funm, stochtrace

    mv, n, dtype = _operator(A)
    if fun is None:
        integrand = stochtrace.monte_carlo_trace()
    else:
        integrand = funm.monte_carlo_funm_sym(funm.dense_funm_sym_eigh(fun), decomp.tridiag_sym(order))
    estimate = stochtrace.estimator_monte_carlo(
        integrand, sampler=stochtrace.sampler_normal(jnp.zeros(n, dtype), num=samples)
    )
    return estimate(mv, _key(key))


def applyfun(A, v, *, fun, order: int = 30, symmetric: bool = True):
    """``f(A)·v``, matrix-free via a Krylov (Lanczos/Arnoldi) approximation — e.g.
    ``fun=lambda z: jnp.exp(-dt*z)`` is one exact exponential-integrator step ``exp(-dt·A)·v``.
    Deterministic (no probes); ``order`` sets the Krylov subspace size.

    ``symmetric=True`` (default) uses **Lanczos** (short recurrence, cheap), assumes ``A = Aᵀ`` (the common
    FEM case), is **differentiable**, and runs on **GPU**. ``symmetric=False`` uses **Arnoldi** for a
    **non-symmetric** operator (advection–diffusion / non-self-adjoint transport): it evaluates ``fun`` on
    the small Hessenberg matrix by a Schur decomposition, so it is **forward-exact for any analytic ``fun``**
    but comes with two limits from ``jax.scipy.linalg.schur`` — it is **CPU-only** (raises on GPU) and
    **not differentiable** (JAX has no ``schur`` derivative). For a **differentiable, GPU** non-symmetric
    time step, use the exponential integrator ``fem.solve(time=jno.solve.exponential())``, which routes the
    non-symmetric matrix exponential through a differentiable Padé approximation instead."""
    _require_matfree()
    from matfree import decomp, funm

    mv, _n, _dtype = _operator(A)
    if symmetric:
        f = funm.funm_lanczos_sym(funm.dense_funm_sym_eigh(fun), decomp.tridiag_sym(order))
    else:  # non-symmetric: Arnoldi Hessenberg + Schur f(H). Forward-exact; Schur blocks the JAX gradient.
        f = funm.funm_arnoldi(funm.dense_funm_schur(fun), decomp.hessenberg(order, reortho="full"))
    return f(mv, jnp.asarray(v))


def diagonal(A, *, fun=None, samples: int = 32, order: int = 25, key=None):
    """Differentiable stochastic estimate of the **diagonal** of ``A`` (Hutchinson), or of ``f(A)`` when
    ``fun`` is given (``A`` symmetric; ``f(A)·probe`` via Lanczos). Unlike :func:`trace` (a scalar) this
    returns the **per-DOF field** — the pointwise version of the same probe estimator. The key use is the
    diagonal of the inverse, ``fun=lambda z: 1/z`` → ``diag(A⁻¹)``: the **pointwise posterior variance /
    uncertainty map** of a FEM precision ``A``, a spatial field you can plot on the mesh.

    Stochastic: an unbiased estimate whose variance falls with ``samples`` and (for ``fun``) whose bias
    falls with ``order`` — **not** exact. Cost is ``samples`` matvecs (``fun=None``) or ``samples×order``
    (``fun`` given, a Lanczos per probe)."""
    _require_matfree()
    from matfree import decomp, funm, stochtrace

    mv, n, dtype = _operator(A)
    if fun is None:
        matvec = mv
    else:  # diag f(A): each probe gets f(A)·probe by Lanczos, then Hutchinson takes the diagonal
        f = funm.funm_lanczos_sym(funm.dense_funm_sym_eigh(fun), decomp.tridiag_sym(order))
        matvec = lambda v: f(mv, v)  # noqa: E731
    estimate = stochtrace.estimator_monte_carlo(
        stochtrace.monte_carlo_diagonal(),
        sampler=stochtrace.sampler_normal(jnp.zeros(n, dtype), num=samples),
    )
    return estimate(matvec, _key(key))
