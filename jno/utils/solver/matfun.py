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
            "jno.solve.logdet / trace / applyfun / diagonal / svd / lstsq need the optional 'matfree' "
            "package "
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


def _raise_krylov_breakdown(who: str, order: int, n: int):
    """Host side of :func:`_finite`. Raises; never returns a value."""
    raise FloatingPointError(
        f"jno.solve.{who}: the Krylov iteration broke down and the result is not finite "
        f"(order={order}, n={n}). Lanczos/Arnoldi can only build a subspace as large as the number of "
        "DISTINCT eigenvalues the probe sees, and a jNO FEM operator has far fewer than it has rows: "
        "every Dirichlet-pinned DOF contributes an identity row, so eigenvalue 1.0 carries a "
        "multiplicity equal to the pinned count. Measured on a 2-D Poisson operator at mesh 0.25 -- "
        "n=30, 16 pinned rows, only 15 distinct eigenvalues -- the default order overran that and this "
        f"returned NaN. Lower order= below the distinct-eigenvalue count (order={max(2, n // 4)} is a "
        "safe start), or apply the estimator to the FREE-DOF operator rather than the pinned one."
    )


def _finite(out, who: str, order: int, n: int):
    """Refuse a Krylov breakdown instead of returning ``NaN``/``inf`` dressed as an estimate.

    ``matfree`` runs its Lanczos/Arnoldi for the full requested ``order`` with no breakdown test, so
    once the Krylov space is exhausted the recurrence normalises by ~0 and the answer degenerates --
    silently, in the middle of an otherwise ordinary call. Checked under a trace as well as eagerly,
    by the same ``lax.cond``-guarded host callback the solver convergence gate uses
    (:func:`jno.utils.solver.solver_api.residual_gate`), because these estimators are meant to be
    differentiated and a check that steps aside under ``grad`` guards nothing.
    """
    arr = jnp.asarray(out)
    ok = jnp.all(jnp.isfinite(arr))
    if not isinstance(ok, jax.core.Tracer):
        if not bool(ok):
            _raise_krylov_breakdown(who, order, n)
        return out
    jax.lax.cond(
        ok,
        lambda: None,
        lambda: jax.debug.callback(_raise_krylov_breakdown, who, order, n),
    )
    return out


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
    return _finite(estimate(mv, _key(key)), "logdet", order, n)


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
    return _finite(estimate(mv, _key(key)), "trace", order, n)


def _dense_funm_eig(fun):
    """A **GPU-capable, differentiable** dense matrix function ``f(H) = V diag(f(λ)) V⁻¹`` for the Arnoldi
    path, replacing ``matfree``'s Schur one (which is CPU-only *and* non-differentiable). The forward uses
    ``jnp.linalg.eig`` (GPU-capable where ``jax.scipy.linalg.schur`` is not); the derivative is supplied
    **analytically** by the Daleckii–Krein / divided-difference formula rather than by differentiating
    through ``eig`` — so it sidesteps JAX's missing non-symmetric-eigenvector derivative. A ``custom_jvp``
    (built only from matmuls, hence transposable ⇒ reverse-mode works too):

        ``L_f(H)[E] = V ( Γ ∘ (V⁻¹ E V) ) V⁻¹``,   ``Γ_ij = (f(λ_i)−f(λ_j))/(λ_i−λ_j)``  (``= f′(λ_i)`` if ``i=j``)

    Requires ``fun`` **holomorphic** (for ``f′`` on the diagonal) and ``H`` **diagonalizable** (generic;
    a defective operator makes ``eig`` ill-conditioned)."""
    fp = jax.grad(fun, holomorphic=True)  # complex derivative for the divided-difference diagonal

    @jax.custom_jvp
    def dense_funm(H):
        real_in = not jnp.iscomplexobj(H)
        lam, V = jnp.linalg.eig(H)
        out = (V * fun(lam)) @ jnp.linalg.inv(V)
        return out.real.astype(H.dtype) if real_in else out

    @dense_funm.defjvp
    def _jvp(primals, tangents):
        (H,), (dH,) = primals, tangents
        real_in = not jnp.iscomplexobj(H)
        lam, V = jnp.linalg.eig(H)
        Vinv = jnp.linalg.inv(V)
        fl = fun(lam)
        out = (V * fl) @ Vinv
        den = lam[:, None] - lam[None, :]
        deg = jnp.abs(den) < 1e-12  # coincident eigenvalues ⇒ divided difference → f′
        gamma = jnp.where(
            deg, jax.vmap(fp)(lam)[:, None] * jnp.ones_like(den), (fl[:, None] - fl[None, :]) / jnp.where(deg, 1.0, den)
        )
        dout = V @ (gamma * (Vinv @ dH.astype(V.dtype) @ V)) @ Vinv
        if real_in:
            out, dout = out.real.astype(H.dtype), dout.real.astype(H.dtype)
        return out, dout

    return dense_funm


def _krylov_funm(mv, v, *, dense_funm, decomposition, order: int):
    """``f(A)·v`` from a Krylov decomposition, stopping at the order that has actually **converged**.

    ``order`` is an upper bound on the Krylov dimension, not an exact request — the standard Krylov
    semantics, and the same meaning ``maxiter`` has for every iterative solver in jNO. That change is
    the fix: run past the dimension the problem supports and the iteration does not gracefully
    saturate, it degrades, silently and catastrophically.

    Measured on a 30-DOF FEM operator whose Dirichlet pinning leaves only 15 distinct eigenvalues
    (``exp(A)·1``, true norm 49.02): the sub-diagonal does not collapse to zero as a textbook "happy
    breakdown" would — it EXPLODES, 0.27 → 2.2 → 4.5 → ... → 184.7, because once the Krylov space is
    exhausted the residual is pure round-off and normalising it yields basis vectors of noise. The
    Ritz values then leave the operator's spectrum entirely (``[-244, +254]`` against a true
    ``[0.99, 5.44]``) and ``f`` is evaluated far outside its intended range:

        order   as computed before      with this rule
           15   rel 3.35e-15            rel 3.26e-15   (k=14)
           20   rel 1.44e-10            rel 3.26e-15   (k=14)
           25   rel 5.11e+35            rel 3.17e-15   (k=14)
           29   rel 1.85e+74            rel 3.17e-15   (k=14)

    Note the order-20 row: this is not only a fix for the catastrophic cases, it is *more accurate*
    wherever round-off has begun to contaminate the basis.

    The rule is the standard a-posteriori one for Krylov ``f(A)v`` — Y. Saad, "Analysis of some Krylov
    subspace approximations to the matrix exponential operator", *SIAM J. Numer. Anal.* **29**(1),
    1992, section 4: accept the first order whose approximation agrees with its predecessor. Two
    things make it nearly free here:

    * every nested approximation comes from the **same** decomposition, so the ladder costs small
      dense ``eigh``/``eig`` calls and **no extra matvecs**;
    * ``Q`` has orthonormal rows, so ``‖y_k − y_{k−1}‖`` in the ``n``-dimensional space equals
      ``‖c_k − c_{k−1}‖`` in the ``k``-dimensional one. The whole test runs in the small space and
      never forms an ``n``-vector, so memory is ``O(order²)`` rather than ``O(order·n)``.

    Measured overhead against the raw ``matfree`` composition it replaces: **0.17 ms at n=513
    (1.24x) and 0.41 ms at n=8355 (1.03x)** — a small constant, because the matvecs dominate as they
    do in any real problem. COMPILE time is where the ladder is not free: one ``lax.cond`` per rung
    grows the jaxpr linearly in ``order``, measured 0.29 s at order=10 to 0.79 s at order=40 against
    a flat ~0.14 s. Under a second either way, and paid once.

    Rungs past the converged one are never EVALUATED, not merely discarded -- see the ``lax.cond``
    below for why zeroing them is not enough to keep the gradient finite.
    """
    # The fourth item's convention DIFFERS between the two decompositions -- `tridiag_sym` returns
    # ||v|| and `hessenberg` returns 1/||v||, which silently scales the Arnoldi answer by ||v||^2
    # (measured: 1.634 against a true 49.020 on a 30-DOF operator, exactly a factor n). Both build
    # Q[0] = v/||v||, so taking the norm here is convention-independent and right for either.
    Q, H, _residual, _norm = decomposition(mv, v)
    norm = jnp.linalg.norm(v)
    m = H.shape[0]
    rtol = 100 * jnp.finfo(H.dtype).eps
    rows, converged, prev, done = [], [], None, jnp.asarray(False)
    for k in range(1, m + 1):
        # Each rung sits behind a `lax.cond` on "nothing has converged yet". Once a rung converges the
        # later ones are never EVALUATED -- which is what keeps the gradient clean, not merely the
        # forward value. Zeroing a bad rung with `where` is not enough: the reverse pass still
        # differentiates the discarded branch and 0 * inf is NaN, and `fun` legitimately overflows on
        # a broken rung (`exp` of a Ritz value at +254). Measured before this: a correct forward with
        # a NaN gradient at order=25. `lax.cond` flows cotangents through the taken branch only.
        #
        # A broken rung can only appear AFTER convergence, never before: at a genuine breakdown the
        # Krylov space is invariant and the approximation is exact there, so consecutive rungs agree
        # and the test fires at or before the last faithful order.
        c_k = jax.lax.cond(
            done,
            lambda: jnp.zeros((k,), H.dtype),
            lambda k=k: jnp.asarray(dense_funm(H[:k, :k])[:, 0], H.dtype),  # f(H_k) e_1
        )
        finite = jnp.all(jnp.isfinite(c_k))
        padded = jnp.zeros((m,), H.dtype).at[:k].set(c_k)
        scale = jnp.linalg.norm(padded)
        ok = (
            jnp.asarray(False)
            if prev is None
            else (~done) & finite & (scale > 0) & (jnp.linalg.norm(padded - prev) <= rtol * scale)
        )
        rows.append(padded)
        converged.append(ok)
        prev, done = padded, done | ok
    C, OK = jnp.stack(rows), jnp.stack(converged)
    # No rung converged: fall back to the full order and let `_finite` speak if it degenerated.
    k_star = jnp.where(jnp.any(OK), jnp.argmax(OK), m - 1)
    return norm * (C[k_star] @ Q)


def applyfun(A, v, *, fun, order: int = 30, symmetric: bool = True):
    """``f(A)·v``, matrix-free via a Krylov (Lanczos/Arnoldi) approximation — e.g.
    ``fun=lambda z: jnp.exp(-dt*z)`` is one exact exponential-integrator step ``exp(-dt·A)·v``.
    Deterministic (no probes); ``order`` **bounds** the Krylov subspace size — the iteration stops at
    the order that has converged (:func:`_krylov_funm`), so raising it can only help. Both paths are
    **differentiable**
    and **GPU-capable**.

    ``symmetric=True`` (default) uses **Lanczos** (short recurrence, cheap) and assumes ``A = Aᵀ`` (the common
    FEM case). ``symmetric=False`` uses **Arnoldi** for a **non-symmetric** operator (advection–diffusion /
    non-self-adjoint transport), with ``f(H)`` on the small Hessenberg matrix computed by an
    eigendecomposition and differentiated analytically (Daleckii–Krein), so it is forward-exact *and*
    reverse-mode differentiable for any **holomorphic** ``fun`` on a **diagonalizable** ``A``."""
    _require_matfree()
    from matfree import decomp, funm

    mv, _n, _dtype = _operator(A)
    if symmetric:
        dense, decomposition = funm.dense_funm_sym_eigh(fun), decomp.tridiag_sym(order)
    else:  # non-symmetric: Arnoldi Hessenberg + eig f(H) — GPU-capable, differentiable (Daleckii–Krein JVP)
        dense, decomposition = _dense_funm_eig(fun), decomp.hessenberg(order, reortho="full")
    out = _krylov_funm(mv, jnp.asarray(v), dense_funm=dense, decomposition=decomposition, order=order)
    return _finite(out, "applyfun", order, _n)


def expmv(A, v, *, order: int = 30):
    """``exp(A)·v`` for a possibly **non-symmetric** ``A``, via Arnoldi + a differentiable **Padé**
    approximation of the small Hessenberg exponential (GPU, reverse-mode differentiable). It is limited to
    the **exponential** (where ``applyfun(..., symmetric=False)`` takes any holomorphic ``fun``), but Padé
    needs **no diagonalizability** — robust on defective/near-defective Hessenbergs where the eig-based
    ``applyfun`` path is ill-conditioned. The engine of the non-symmetric exponential time integrator; scale
    ``A`` into its matvec to get ``exp(dt·A)·v``."""
    _require_matfree()
    from matfree import decomp, funm

    mv, _n, _dtype = _operator(A)
    out = _krylov_funm(
        mv,
        jnp.asarray(v),
        dense_funm=funm.dense_funm_pade_exp(),
        decomposition=decomp.hessenberg(order, reortho="full"),
        order=order,
    )
    return _finite(out, "expmv", order, _n)


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
    return _finite(estimate(matvec, _key(key)), "diagonal", order, n)


def svd(A, *, k: int = 6, depth: int | None = None, v0=None):
    r"""Differentiable, matrix-free **partial SVD** ``A ≈ U diag(s) Vᵀ`` — the ``k`` largest singular
    triplets of a possibly **rectangular** operator, via Golub–Kahan bidiagonalization.

    G. Golub & W. Kahan, "Calculating the Singular Values and Pseudo-Inverse of a Matrix",
    *J. SIAM Numer. Anal. Ser. B* 2(2), 1965 — the bidiagonalization whose singular values are the
    Ritz approximations to those of ``A``.

    Complements :func:`jno.solve.eigs`, which solves the *symmetric* generalized eigenproblem
    ``Kx = λMx``. The SVD is the tool for the two things that are not eigenproblems:

    * **POD / reduced-order models** — the left singular vectors of a snapshot matrix are the
      energy-optimal basis, and ``s`` says how many modes the trajectory actually needs.
    * **Ill-posedness of an inverse problem** — the singular spectrum of the parameter-to-observable
      map says which parameter modes are recoverable at all; the ones under the noise floor are not,
      no matter the optimizer. Since ``A`` is only ever touched through its matvec, that map can be a
      JVP of a differentiable FEM solve — never assembled.

    ``depth`` is the number of bidiagonalization steps (default ``2k + 10``, capped at
    ``min(m, n)``). **It must exceed ``k``**: the Ritz values converge from below, so at
    ``depth == k`` only the largest singular value is meaningful — measured 95% error on the rest,
    against ~1e-15 at ``depth = 2k`` on the same operator. Convergence is fast for the *decaying*
    spectra that make POD and ill-posedness analysis worth doing, and slow for clustered ones (a
    tightly clustered spectrum still showed ~3% error at ``depth = 4k``) — check ``s`` for a
    plateau if the spectrum may be flat.

    Returns ``(U, s, Vt)`` with ``U`` ``(m, k)``, ``s`` ``(k,)`` descending, ``Vt`` ``(k, n)``.
    """
    _require_matfree()
    from matfree import decomp, eig

    m, n = A.shape
    mv = A.mv if hasattr(A, "mv") else (lambda v: A @ v)
    dtype = jax.eval_shape(mv, jnp.zeros(n)).dtype
    p = int(min(m, n))
    k = int(k)
    if k < 1 or k > p:
        raise ValueError(f"jno.solve.svd: k={k} must be in 1..min(m, n)={p} for an operator of shape {(m, n)}.")
    d = int(depth) if depth is not None else min(2 * k + 10, p)
    if d < k:
        raise ValueError(
            f"jno.solve.svd: depth={d} is below k={k} — the bidiagonalization cannot resolve more "
            "singular values than it takes steps. Use depth >= k (the default 2k+10 oversamples, "
            "which is what makes the smaller singular values accurate)."
        )
    d = min(d, p)
    v0 = jnp.ones((n,), dtype) / jnp.sqrt(jnp.asarray(n, dtype)) if v0 is None else jnp.asarray(v0).reshape(-1)
    # matfree derives the transpose action by transposing the (linear) matvec, so only `mv` is needed.
    U, s, Vt = eig.svd_partial(decomp.bidiag(d, materialize=True))(mv, v0)
    # matfree returns the bases row-wise ((depth, m) / (depth, n)); truncate to k and orient U as (m, k)
    return jnp.asarray(U)[:k].T, jnp.asarray(s)[:k], jnp.asarray(Vt)[:k]


def lstsq(A, b, *, damp: float = 0.0, atol: float = 1e-6, btol: float = 1e-6, maxiter: int = 100_000, x0=None):
    """Differentiable, matrix-free **least-squares** ``min_x ‖A x − b‖²`` for a **rectangular** ``A``
    (over- or under-determined), via LSMR — the gap left by the square ``Ax=b`` solvers. ``damp`` adds
    Tikhonov regularisation ``+ damp²‖x‖²`` (well-posed for rank-deficient / ill-posed inverse problems);
    ``x0`` an initial guess. ``A`` may be a dense/sparse matrix or a jNO ``LinearOperator`` (only its
    matvec and transpose are used — matrix-free). Returns the solution ``x``.

    Scope: **real** operators (LSMR's complex path is untested here). Convergence is controlled by
    ``atol``/``btol``/``maxiter`` — a stalled solve returns its best iterate, so check the residual."""
    _require_matfree()
    from matfree import lstsq as _lstsq

    m, n = A.shape
    mv = A.mv if hasattr(A, "mv") else (lambda v: A @ v)
    dtype = jax.eval_shape(mv, jnp.zeros(n)).dtype
    # LSMR wants the vector–matrix product v ↦ vᵀA; get it as the transpose of the matvec (matrix-free)
    vecmat = lambda v: jax.linear_transpose(mv, jnp.zeros(n, dtype))(v)[0]  # noqa: E731
    solve = _lstsq.lsmr(atol=atol, btol=btol, maxiter=maxiter)
    out = solve(vecmat, jnp.asarray(b), x0=x0, damp=damp)
    return out[0] if isinstance(out, tuple) else out
