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

__all__ = ["fgmres", "minres", "chebyshev_iteration", "power_iteration_bound"]

_TINY = 1e-300


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
    bnorm = jnp.linalg.norm(b)
    tol_abs = tol * jnp.maximum(bnorm, _TINY)
    max_cycles = -(-int(maxiter) // m)  # ceil

    def cycle(x):
        r0 = b - matvec(x)
        beta = jnp.linalg.norm(r0)
        V = jnp.zeros((m + 1, n), b.dtype).at[0].set(r0 / jnp.maximum(beta, _TINY))
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
            h = (V @ w) * mask
            w = w - h @ V
            h2 = (V @ w) * mask
            w = w - h2 @ V
            h = h + h2
            hj1 = jnp.linalg.norm(w)
            hcol = h.at[j + 1].set(hj1)

            # apply the stored Givens rotations of columns 0..j-1 to the new column
            def rot(i, hc):
                hi, hi1 = hc[i], hc[i + 1]
                new_i = cs[i] * hi + sn[i] * hi1
                new_i1 = -sn[i] * hi + cs[i] * hi1
                return jnp.where(i < j, hc.at[i].set(new_i).at[i + 1].set(new_i1), hc)

            hcol = jax.lax.fori_loop(0, m, rot, hcol)

            # new rotation annihilating the subdiagonal entry
            denom = jnp.sqrt(hcol[j] ** 2 + hcol[j + 1] ** 2)
            c_new = hcol[j] / jnp.maximum(denom, _TINY)
            s_new = hcol[j + 1] / jnp.maximum(denom, _TINY)
            hcol = hcol.at[j].set(denom).at[j + 1].set(0.0)
            gj = g[j]

            new = (
                V.at[j + 1].set(w / jnp.maximum(hj1, _TINY)),
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

    r1 = b - matvec(x0)
    y = M(r1)
    beta1 = jnp.sqrt(jnp.maximum(r1 @ y, 0.0))
    tol_abs = tol * jnp.maximum(beta1, _TINY)

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
        v = y / jnp.maximum(beta, _TINY)
        y = matvec(v)
        y = y - jnp.where(itn >= 1, beta / jnp.maximum(oldb, _TINY), 0.0) * r1
        alfa = v @ y
        y = y - (alfa / jnp.maximum(beta, _TINY)) * r2
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
        gamma = jnp.maximum(jnp.sqrt(gbar**2 + beta**2), _TINY)
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
        return w / jnp.maximum(jnp.linalg.norm(w), _TINY), lam

    _, lam = jax.lax.fori_loop(0, iters, step, (v0, jnp.asarray(1.0, dtype)))
    return jnp.abs(lam)


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
    bnorm = jnp.linalg.norm(b)
    tol_abs = tol * jnp.maximum(bnorm, _TINY)

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
