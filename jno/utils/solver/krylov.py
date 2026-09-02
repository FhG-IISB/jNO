"""Pure-JAX Krylov solvers absent from the JAX ecosystem: FGMRES, MINRES, Chebyshev.

These are the *only* iteration loops jNO implements itself (design rule: reuse
``jax.scipy.sparse.linalg`` / ``sparse_lu_solve`` for everything that exists upstream — see
``plans/fem-solver-api.md``). Each is textbook-grade with a published pseudocode origin, cited in
its docstring and in ``docs/fem.md``, and is pinned against a scipy/dense oracle in
``tests/test_fem_solver_krylov.py``.

All three are fixed-shape ``lax.while_loop``/``fori_loop`` implementations: ``jit``- and
``vmap``-native, GPU-friendly (matvecs and dense small-array work only). Differentiability is
added one level up (``jno.solve``) via ``lax.custom_linear_solve``, so the loops themselves are
never differentiated.

Conventions: ``matvec`` is ``v -> A v``; ``M`` is the ``v -> M^{-1} v`` applier (identity when
``None``); ``x0`` the initial guess (zeros when ``None``); ``tol`` is relative to ``||b||``
(``atol=0`` semantics, matching ``jax.scipy``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = [
    "fgmres",
    "minres",
    "cocg",
    "chebyshev_iteration",
    "power_iteration_bound",
    "lanczos_spectrum_bounds",
    "spectrum_bounds",
]


def _EPS_OF(dtype):
    return jnp.finfo(jnp.dtype(dtype)).eps


def _tiny_of(dtype):
    """Smallest positive normal of ``dtype`` — the divide-by-zero floor for the breakdown guards.

    One hard-coded constant cannot serve both precisions. The historical ``1e-300`` **underflows to
    exactly 0.0 in float32** (whose smallest normal is ~1.18e-38), so every ``maximum(x, tiny)`` guard
    below degenerated into ``maximum(x, 0.0)`` and divided by exact zero on breakdown, yielding
    ``inf``/``NaN`` instead of a clamped value. Reciprocals stay finite either way: ``1/tiny`` is
    ~8.5e37 (float32) and ~4.5e307 (float64), both representable.
    """
    return jnp.finfo(jnp.dtype(dtype)).tiny


def _effective_tol(tol, dtype, *, floor_factor=4.0):
    """The relative tolerance ``dtype`` can actually reach.

    A tolerance below unit round-off is unsatisfiable: the residual norm cannot fall below
    ``~eps*||b||``, so the ``while_loop`` runs to ``maxiter`` on a system it has already solved. The
    shipped default ``tol=1e-8`` is exactly such a request in float32 (eps 1.2e-7).

    Floored at a small multiple of eps. **float64 is untouched** for any tolerance anyone passes in
    practice (the floor there is ~8.9e-16), so this changes no existing result.
    """
    return float(max(float(tol), floor_factor * float(_EPS_OF(dtype))))


def _ident(v):
    return v


# ---------------------------------------------------------------------------
# FGMRES
# ---------------------------------------------------------------------------


def fgmres(matvec, b, *, M=None, x0=None, tol=1e-8, restart=30, maxiter=1000):
    """Flexible restarted GMRES — right preconditioning with a per-iteration-varying ``M``.

    Y. Saad, *A Flexible Inner-Outer Preconditioned GMRES Algorithm*, SIAM J. Sci. Stat.
    Comput. 14(2), 1993, Algorithm 2.2. Unlike plain GMRES, the preconditioned directions
    ``z_j = M_j^{-1} v_j`` are stored, so ``M`` may itself be an *iterative* solve (inner Krylov,
    multigrid-with-tolerance) — the standard outer solver for block/Schur-preconditioned saddle
    systems. Orthogonalisation is classical Gram-Schmidt with one reorthogonalisation (CGS2):
    fully vectorised (two ``(m+1, n)`` matvecs per step), the JAX-friendly substitute for
    sequential MGS at equal numerical quality.

    Memory: two ``(restart, n)`` bases. Each restart cycle runs its ``restart`` inner steps at
    fixed shape (converged/broken-down steps become masked no-ops); the outer loop re-forms the
    true residual, so a masked cycle costs work but never accuracy.
    """
    M = M or _ident
    b = jnp.asarray(b)
    n = b.shape[0]
    m = int(min(restart, n))
    x0 = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0).reshape(-1)
    tiny = _tiny_of(b.dtype)  # float32-safe breakdown floor (see _tiny_of)
    tol = _effective_tol(tol, b.dtype)  # a sub-eps request cannot converge; floor it
    bnorm = jnp.linalg.norm(b)
    tol_abs = tol * jnp.maximum(bnorm, tiny)
    max_cycles = -(-int(maxiter) // m)  # ceil

    def cycle(x):
        r0 = b - matvec(x)
        beta = jnp.linalg.norm(r0)
        V = jnp.zeros((m + 1, n), b.dtype).at[0].set(r0 / jnp.maximum(beta, tiny))
        Z = jnp.zeros((m, n), b.dtype)
        H = jnp.zeros((m + 1, m), b.dtype)
        g = jnp.zeros((m + 1,), b.dtype).at[0].set(beta)
        cs = jnp.zeros((m,), b.dtype)
        sn = jnp.zeros((m,), b.dtype)

        def step(j, carry):
            V, Z, H, g, cs, sn, active = carry
            z = M(V[j])
            w = matvec(z)
            # CGS2 against v_0..v_j (mask selects the built part of the basis)
            mask = (jnp.arange(m + 1) <= j).astype(b.dtype)
            # HERMITIAN inner products: Arnoldi orthogonalises in <x, y> = xᴴy, so the basis must be
            # conjugated. `V @ w` is the bilinear form -- correct on reals, wrong on complex, where it
            # silently produces a non-orthogonal basis. `jnp.conj` on a real array is a no-op.
            h = (jnp.conj(V) @ w) * mask
            w = w - h @ V
            h2 = (jnp.conj(V) @ w) * mask
            w = w - h2 @ V
            h = h + h2
            hj1 = jnp.linalg.norm(w)
            hcol = h.at[j + 1].set(hj1)

            # apply the stored Givens rotations of columns 0..j-1 to the new column
            def rot(i, hc):
                hi, hi1 = hc[i], hc[i + 1]
                new_i = jnp.conj(cs[i]) * hi + jnp.conj(sn[i]) * hi1
                new_i1 = -sn[i] * hi + cs[i] * hi1
                return jnp.where(i < j, hc.at[i].set(new_i).at[i + 1].set(new_i1), hc)

            hcol = jax.lax.fori_loop(0, m, rot, hcol)

            # new rotation annihilating the subdiagonal entry
            # Givens on MAGNITUDES, not squares: `h**2` is negative-capable and complex-wrong, so the
            # rotation has to be built from |h|. `c` comes out real and non-negative and `s` carries the
            # phase of h[j], which is what makes the rotation unitary rather than merely orthogonal.
            # On real input this is the same rotation up to the sign convention (c >= 0 always), and a
            # sign flip of a Givens pair leaves the least-squares solution -- hence `y` -- unchanged.
            aj = jnp.abs(hcol[j])
            denom = jnp.sqrt(aj**2 + jnp.abs(hcol[j + 1]) ** 2)
            safe = jnp.maximum(denom, tiny)
            c_new = (aj / safe).astype(b.dtype)
            phase = jnp.where(aj > tiny, hcol[j] / jnp.maximum(aj, tiny), jnp.ones((), b.dtype))
            s_new = hcol[j + 1] * jnp.conj(phase) / safe
            hcol = hcol.at[j].set(denom * phase).at[j + 1].set(0.0)
            gj = g[j]

            new = (
                V.at[j + 1].set(w / jnp.maximum(hj1, tiny)),
                Z.at[j].set(z),
                H.at[:, j].set(hcol),
                g.at[j].set(c_new * gj).at[j + 1].set(-s_new * gj),
                cs.at[j].set(c_new),
                sn.at[j].set(s_new),
                active & (jnp.abs(s_new * gj) > tol_abs),  # |g[j+1]| is the residual norm
            )
            old = (V, Z, H, g, cs, sn, active)
            return jax.tree_util.tree_map(lambda a, o: jnp.where(active, a, o), new, old)

        active0 = beta > tol_abs
        V, Z, H, g, cs, sn, _ = jax.lax.fori_loop(0, m, step, (V, Z, H, g, cs, sn, active0))

        # never-written columns are zero: unit diagonal keeps the triangular solve regular,
        # and their (spurious) y entries multiply the zero rows of Z — harmless by construction
        Hm = H[:m, :m]
        written = jnp.abs(jnp.diagonal(Hm)) > 0.0
        Hm = Hm + jnp.diag(jnp.where(written, 0.0, 1.0))
        y = jax.scipy.linalg.solve_triangular(Hm, g[:m], lower=False)
        return x + y @ Z

    def cond(state):
        x, k = state
        return (jnp.linalg.norm(b - matvec(x)) > tol_abs) & (k < max_cycles)

    x, _ = jax.lax.while_loop(cond, lambda s: (cycle(s[0]), s[1] + 1), (x0, 0))
    return x


# ---------------------------------------------------------------------------
# MINRES
# ---------------------------------------------------------------------------


def minres(matvec, b, *, M=None, x0=None, tol=1e-8, maxiter=2000):
    """MINRES — **symmetric** (possibly indefinite) systems: saddle points, Helmholtz-like shifts.

    C. C. Paige & M. A. Saunders, *Solution of Sparse Indefinite Systems of Linear Equations*,
    SIAM J. Numer. Anal. 12(4), 1975, §5 (the Lanczos + Givens ``QR`` recurrence; state layout
    follows the classic reference implementation, e.g. ``scipy.sparse.linalg.minres``).
    The preconditioner ``M`` must be symmetric **positive definite** even when ``A`` is
    indefinite (the Lanczos inner products are ``M^{-1}``-weighted); convergence is measured on
    the ``M^{-1}``-norm residual estimate ``phibar`` relative to its initial value.
    """
    M = M or _ident
    b = jnp.asarray(b)
    x0 = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0).reshape(-1)

    tiny = _tiny_of(b.dtype)  # float32-safe breakdown floor (see _tiny_of)
    tol = _effective_tol(tol, b.dtype)  # a sub-eps request cannot converge; floor it

    r1 = b - matvec(x0)
    y = M(r1)
    beta1 = jnp.sqrt(jnp.maximum(r1 @ y, 0.0))
    tol_abs = tol * jnp.maximum(beta1, tiny)

    # state: x, r1, r2, y, oldb, beta, dbar, epsln, phibar, cs, sn, w, w2, itn
    zeros = jnp.zeros_like(b)
    state0 = (
        x0,
        r1,
        r1,
        y,
        jnp.array(0.0, b.dtype),
        beta1,
        jnp.array(0.0, b.dtype),
        jnp.array(0.0, b.dtype),
        beta1,
        jnp.array(-1.0, b.dtype),
        jnp.array(0.0, b.dtype),
        zeros,
        zeros,
        0,
    )

    def cond(s):
        phibar, itn = s[8], s[13]
        return (phibar > tol_abs) & (itn < maxiter)

    def body(s):
        x, r1, r2, y, oldb, beta, dbar, epsln, phibar, cs, sn, w, w2, itn = s
        v = y / jnp.maximum(beta, tiny)
        y = matvec(v)
        y = y - jnp.where(itn >= 1, beta / jnp.maximum(oldb, tiny), 0.0) * r1
        alfa = v @ y
        y = y - (alfa / jnp.maximum(beta, tiny)) * r2
        r1, r2 = r2, y
        y = M(r2)
        oldb, beta = beta, jnp.sqrt(jnp.maximum(r2 @ y, 0.0))

        # previous rotation
        oldeps = epsln
        delta = cs * dbar + sn * alfa
        gbar = sn * dbar - cs * alfa
        epsln = sn * beta
        dbar = -cs * beta
        # next rotation
        gamma = jnp.maximum(jnp.sqrt(gbar**2 + beta**2), tiny)
        cs, sn = gbar / gamma, beta / gamma
        phi = cs * phibar
        phibar = sn * phibar
        # solution update
        w1, w2 = w2, w
        w = (v - oldeps * w1 - delta * w2) / gamma
        x = x + phi * w
        return x, r1, r2, y, oldb, beta, dbar, epsln, phibar, cs, sn, w, w2, itn + 1

    return jax.lax.while_loop(cond, body, state0)[0]


# ---------------------------------------------------------------------------
# COCG
# ---------------------------------------------------------------------------


def cocg(matvec, b, *, M=None, x0=None, tol=1e-8, maxiter=2000):
    """COCG — **complex-symmetric** systems, ``A = A^T`` (NOT ``A = A^H``).

    H. A. van der Vorst & J. B. M. Melissen, *A Petrov-Galerkin type method for solving Ax = b,
    where A is symmetric complex*, IEEE Trans. Magn. 26(2), 1990, 706-708.

    The time-harmonic operators assembled here — eddy-current A-V, Helmholtz, RCWA — are complex
    **symmetric** rather than Hermitian. For those, CG's three-term recurrence survives if the
    Hermitian inner product is replaced by the **bilinear** form ``x^T y``, giving a short
    recurrence where GMRES needs a growing basis: one matvec, one preconditioner apply and O(n)
    work per iteration, with no restart parameter to tune.

    **The inner product here is bilinear, and that is deliberate.** ``fgmres`` above conjugates its
    Arnoldi projections (``jnp.conj(V) @ w``) because it orthogonalises in the Hermitian inner
    product; COCG must NOT, and ``@`` on 1-D arrays is exactly the non-conjugating contraction the
    method wants. Adding a ``conj`` here to "match fgmres" silently turns this into a method for a
    matrix that is not the one being solved.

    Two consequences of the bilinear form, both handled below:

      * ``x^T x`` is not a norm — it can vanish for ``x != 0`` — so **convergence is still measured
        in the ordinary Hermitian 2-norm** ``||r||`` relative to ``||b||``. Only the projections
        are bilinear.
      * the method can **break down** (``p^T A p = 0`` with ``p != 0``) where CG on an SPD system
        cannot. That is a genuine property, not a rounding artefact, so it stops the iteration and
        returns the last iterate rather than dividing by a clamped denominator and reporting a
        confident wrong answer. ``jno.solve``'s residual guard then reports it.

    ``M`` is the ``v -> M^{-1} v`` applier and should itself be complex-symmetric for the recurrence
    to remain valid.
    """
    M = M or _ident
    b = jnp.asarray(b)
    x0 = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0).reshape(-1)

    rdtype = jnp.zeros((), b.dtype).real.dtype  # norms and tolerances are REAL even for complex b
    tiny = _tiny_of(rdtype)  # float32-safe breakdown floor (see _tiny_of)
    tol = _effective_tol(tol, rdtype)  # a sub-eps request cannot converge; floor it

    r0 = b - matvec(x0)
    z0 = M(r0)
    tol_abs = tol * jnp.maximum(jnp.linalg.norm(b), tiny)

    # state: x, r, z, p, rho, itn, broke
    state0 = (x0, r0, z0, z0, r0 @ z0, jnp.asarray(0), jnp.asarray(False))

    def cond(s):
        r, itn, broke = s[1], s[5], s[6]
        return (jnp.linalg.norm(r) > tol_abs) & (itn < maxiter) & jnp.logical_not(broke)

    def body(s):
        x, r, z, p, rho, itn, broke = s
        q = matvec(p)
        pq = p @ q  # bilinear, NOT conj(p) @ q — see the docstring
        bad = (jnp.abs(pq) <= tiny) | (jnp.abs(rho) <= tiny)
        # Freeze the iterate on breakdown (alpha = beta = 0) instead of propagating inf/NaN, so the
        # returned x is the last good one and the caller's residual check sees the truth.
        alpha = jnp.where(bad, 0.0, rho / jnp.where(bad, 1.0, pq))
        x = x + alpha * p
        r = r - alpha * q
        z = M(r)
        rho_new = r @ z
        beta = jnp.where(bad, 0.0, rho_new / jnp.where(bad, 1.0, rho))
        p = z + beta * p
        return (x, r, z, p, rho_new, itn + 1, broke | bad)

    return jax.lax.while_loop(cond, body, state0)[0]


# ---------------------------------------------------------------------------
# Chebyshev
# ---------------------------------------------------------------------------


def power_iteration_bound(matvec, n, *, dtype=None, iters=30, M=None):
    """Largest-eigenvalue estimate of ``M^{-1} A`` by power iteration (deterministic start).

    ``iters`` fixed steps on a normalized ones-vector; returns the final Rayleigh-quotient
    magnitude. Cheap (one matvec per step), ``jit``/``vmap``-native, good to a few percent on
    the FEM operators this preconditions — inflate by a safety factor at the call site.
    """
    M = M or _ident
    dtype = dtype or jnp.result_type(float)
    v0 = jnp.ones((n,), dtype) / jnp.sqrt(jnp.asarray(n, dtype))

    def step(_, carry):
        v, _lam = carry
        w = M(matvec(v))
        lam = v @ w
        return w / jnp.maximum(jnp.linalg.norm(w), _tiny_of(dtype)), lam

    _, lam = jax.lax.fori_loop(0, iters, step, (v0, jnp.asarray(1.0, dtype)))
    return jnp.abs(lam)


def lanczos_spectrum_bounds(matvec, n, *, dtype=None, iters=30, M=None):
    """**Both** ends of the spectrum of ``M^{-1} A`` (SPD) from one Krylov space, via Lanczos.

    C. Lanczos, "An iteration method for the solution of the eigenvalue problem of linear
    differential and integral operators", *J. Res. Natl. Bur. Stand.* 45(4), 1950, §II — the
    symmetric tridiagonalization whose extreme **Ritz values** (the eigenvalues of the ``k x k``
    tridiagonal ``T_k``) bound the extreme eigenvalues of ``A`` from the inside, converging to
    them from within as ``k`` grows.

    This exists because power iteration gives only ``lmax``. A Chebyshev polynomial is optimal on
    the interval it is fitted to, so a fabricated ``lmin`` (the historical ``lmax / 30`` guess)
    fits the wrong interval and degrades the preconditioner — badly when the true ratio is far
    from the guess. Lanczos returns both ends for the same one-matvec-per-step cost.

    Returns ``(lmin, lmax)``, or ``None`` when :mod:`matfree` is not installed (an OPTIONAL
    dependency) or the decomposition is degenerate — the caller then falls back to power
    iteration. Both ends are Ritz values, hence *interior* to the true spectrum, so the caller
    should still inflate ``lmax`` by a safety factor and deflate ``lmin``.
    """
    try:
        from matfree import decomp
    except ImportError:  # optional dependency: the caller falls back to power_iteration_bound
        return None

    M = M or _ident
    dtype = dtype or jnp.result_type(float)
    k = int(min(max(int(iters), 2), max(2, n - 1)))  # Lanczos needs 2 <= num_matvecs <= n-1
    v0 = jnp.ones((n,), dtype) / jnp.sqrt(jnp.asarray(n, dtype))
    try:
        # tridiag_sym(k) builds the decomposition; `J_small` is the materialized k x k
        # tridiagonal, so its eigenvalues are the Ritz values directly. k is the Lanczos depth
        # (~30), so this dense eigenproblem is trivial — nothing the size of the FEM operator.
        out = decomp.tridiag_sym(k)(lambda v: M(matvec(v)), v0)
        vals = jnp.linalg.eigvalsh(jnp.asarray(out.J_small))
        lo, hi = jnp.min(vals), jnp.max(vals)
    except Exception:  # a degenerate / broken-down decomposition must not fail the whole solve
        return None
    # A breakdown can collapse or invert the interval; reject rather than hand the Chebyshev
    # recurrence something it divides by (delta = (hi - lo)/2 must be > 0).
    if not (jnp.isfinite(lo) and jnp.isfinite(hi)) or hi <= 0.0 or hi - lo <= 0.0:
        return None
    return jnp.abs(lo), jnp.abs(hi)


def nystrom_sketch(matvec, n, *, rank, key, dtype=None):
    r"""Randomized Nyström approximation ``A ≈ U diag(lam) U^T`` of an SPD operator, from matvecs.

    Frangella, Tropp & Udell, "Randomized Nyström Preconditioning", *SIAM J. Matrix Anal. Appl.*
    44(2), 2023, Algorithm 2.1 (the sketch) — the stabilized construction, which avoids the
    catastrophic cancellation of the naive ``Y (Ω^T Y)^{-1} Y^T`` form.

    Costs exactly ``rank`` matvecs: the sketch ``Y = A Ω`` is a single batched matvec against an
    ``n x rank`` Gaussian test matrix. Everything after that is dense work on ``n x rank`` and
    ``rank x rank`` factors, so nothing the size of ``A`` is ever formed.

    Returns ``(U, lam)`` with ``U`` orthonormal ``(n, rank)`` and ``lam`` the ``rank`` non-negative
    approximate eigenvalues, largest first.
    """
    dtype = dtype or jnp.result_type(float)
    k = int(min(max(int(rank), 1), n))
    omega = jax.random.normal(key, (n, k), dtype)
    omega, _ = jnp.linalg.qr(omega)  # orthonormal test matrix: better conditioned sketch
    Y = jax.vmap(matvec, in_axes=1, out_axes=1)(omega)  # rank matvecs -> (n, k)

    # Stabilization shift nu (Alg. 2.1): lifts Omega^T Y_nu to be safely positive definite so the
    # Cholesky below cannot fail on a numerically semi-definite sketch.
    nu = jnp.sqrt(jnp.asarray(n, dtype)) * _EPS_OF(dtype) * jnp.linalg.norm(Y)
    Y_nu = Y + nu * omega
    C = jnp.linalg.cholesky(omega.T @ Y_nu)
    # B = Y_nu C^{-T} via a triangular solve, so A ~ B B^T without inverting anything
    B = jax.scipy.linalg.solve_triangular(C, Y_nu.T, lower=True).T
    U, s, _ = jnp.linalg.svd(B, full_matrices=False)
    lam = jnp.maximum(s**2 - nu, 0.0)  # undo the shift; clamp at 0 (the approximation is PSD)
    return U, lam


def nystrom_apply(U, lam, mu):
    r"""The Nyström **preconditioner** application ``P^{-1} v`` for ``A ≈ U diag(lam) U^T``.

    Frangella, Tropp & Udell 2023, §3 (Definition 3.1)::

        P^{-1} = (lam_min + mu) · U (diag(lam) + mu I)^{-1} U^T  +  (I - U U^T)

    The low-rank part deflates the captured (largest) eigenvalues towards 1 while the orthogonal
    complement is left alone, so ``P^{-1} A`` has its top of the spectrum flattened — which is
    exactly the part Jacobi cannot touch. ``mu`` is the regularization / smallest retained
    eigenvalue; larger ``mu`` is a weaker but safer preconditioner.
    """
    lam_min = lam[-1]
    scale = (lam_min + mu) / (lam + mu)

    def apply(v):
        c = U.T @ v
        return U @ (scale * c) + (v - U @ c)

    return apply


def spectrum_bounds(matvec, n, *, dtype=None, iters=30, M=None, lmin=None, lmax=None, safety=1.05, lmin_ratio=1.0 / 30.0):
    """The ``(lmin, lmax)`` a Chebyshev recurrence should be fitted to — the single place both
    ``jno.solve.chebyshev`` and ``jno.precond.chebyshev`` decide it.

    Caller-supplied bounds always win. Otherwise prefer :func:`lanczos_spectrum_bounds`, which
    measures **both** ends, and fall back to :func:`power_iteration_bound` (``lmax`` only, with
    ``lmin = lmin_ratio * lmax``) when :mod:`matfree` is absent or the decomposition breaks down.

    Why this matters: the Chebyshev polynomial is *optimal on the interval it is given*, and is
    only a contraction inside it. A fabricated ``lmin`` that lands above the true smallest
    eigenvalue leaves the modes below it outside the fitted interval, where the polynomial
    **amplifies** them instead of damping. The fallback's ``lmax/30`` is a smoother-style guess
    that is safe when the true ratio is smaller than it assumes and harmful when it is not.

    The Ritz values are interior to the true spectrum, so ``lmax`` is still inflated by ``safety``
    and ``lmin`` deflated by the same factor to cover the ends Lanczos has not yet reached.
    """
    if lmin is not None and lmax is not None:
        return float(lmin), float(lmax)
    est = None if lmax is not None else lanczos_spectrum_bounds(matvec, n, dtype=dtype, iters=iters, M=M)
    if est is not None:
        lo_e, hi_e = est
        hi = safety * hi_e
        lo = lmin if lmin is not None else lo_e / safety
        return lo, hi
    hi = lmax if lmax is not None else safety * power_iteration_bound(matvec, n, dtype=dtype, iters=iters, M=M)
    lo = lmin if lmin is not None else lmin_ratio * hi
    return lo, hi


def chebyshev_iteration(matvec, b, *, lmin, lmax, M=None, x0=None, tol=1e-8, maxiter=200):
    """Chebyshev semi-iteration for SPD systems with spectrum inside ``[lmin, lmax]``.

    Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM 2003, §12.3,
    Algorithm 12.1 (three-term recurrence; the method originates with Golub & Varga 1961).
    **Inner-product free** — only matvecs and AXPYs — which is what makes it the GPU-era
    smoother/preconditioner (no reductions, trivially vmappable) and a fixed *linear* operator
    in ``b`` for fixed step count (so it can precondition CG). ``M`` (SPD) preconditions the
    recurrence; ``lmin``/``lmax`` then bound the spectrum of ``M^{-1} A``. Convergence degrades
    gracefully when the bounds are loose — prefer a small safety margin on ``lmax``.
    """
    M = M or _ident
    b = jnp.asarray(b)
    x = jnp.zeros_like(b) if x0 is None else jnp.asarray(x0).reshape(-1)
    theta = 0.5 * (lmax + lmin)
    delta = 0.5 * (lmax - lmin)
    sigma1 = theta / delta
    tiny = _tiny_of(b.dtype)  # float32-safe breakdown floor (see _tiny_of)
    tol = _effective_tol(tol, b.dtype)  # a sub-eps request cannot converge; floor it
    bnorm = jnp.linalg.norm(b)
    tol_abs = tol * jnp.maximum(bnorm, tiny)

    r = b - matvec(x)
    d = M(r) / theta
    rho = 1.0 / sigma1

    def cond(s):
        _x, r, _d, _rho, k = s
        return (jnp.linalg.norm(r) > tol_abs) & (k < maxiter)

    def body(s):
        x, r, d, rho, k = s
        x = x + d
        r = r - matvec(d)
        rho_new = 1.0 / (2.0 * sigma1 - rho)
        d = rho_new * rho * d + (2.0 * rho_new / delta) * M(r)
        return x, r, d, rho_new, k + 1

    return jax.lax.while_loop(cond, body, (x, r, d, rho, 0))[0]


def chebyshev_apply(matvec, v, *, lmin, lmax, degree, M=None):
    """Fixed-``degree`` Chebyshev polynomial application ``p(A) v ≈ A^{-1} v`` (no convergence
    test — a *linear* operator in ``v``, usable as a preconditioner for CG/MINRES/FGMRES)."""
    M = M or _ident
    theta = 0.5 * (lmax + lmin)
    delta = 0.5 * (lmax - lmin)
    sigma1 = theta / delta
    x = jnp.zeros_like(v)
    r = v  # x = 0 start
    d = M(r) / theta
    rho = 1.0 / sigma1

    def body(_, s):
        x, r, d, rho = s
        x = x + d
        r = r - matvec(d)
        rho_new = 1.0 / (2.0 * sigma1 - rho)
        d = rho_new * rho * d + (2.0 * rho_new / delta) * M(r)
        return x, r, d, rho_new

    return jax.lax.fori_loop(0, degree, body, (x, r, d, rho))[0]


__all__.append("chebyshev_apply")
