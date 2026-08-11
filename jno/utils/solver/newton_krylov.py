"""Jacobian-free Newton-Krylov root find -- the optimistix-free nonlinear default.

Solves ``residual_fn(u) = 0`` with Newton's method whose per-step linear solve
``J delta = -R`` is done matrix-free: ``J @ v`` comes from a JVP (``jax.linearize``)
so the Jacobian is never formed, and the solve is BiCGStab (handles non-symmetric
``J``). Implicit differentiation is provided by ``jax.lax.custom_root`` (so gradients
reach parameters closed over by ``residual_fn`` without unrolling Newton), and the
inner BiCGStab is wrapped in ``jax.lax.custom_linear_solve`` so the reverse pass --
which solves ``J^T`` -- gets an analytic transpose rule instead of trying to
differentiate the solver's ``while_loop``.

No external solver dependency (no optimistix / lineax).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

__all__ = ["newton_krylov", "newton_direct", "staggered_newton", "bicgstab"]

_EPS = 1e-300


def _convergence_check(f0, u0, u, *, rtol, atol, max_steps, who):
    """Raise (eagerly) if the Newton loop returned on its STEP CAP rather than on the tolerance.

    Both drivers below iterate a ``jax.lax.while_loop`` whose condition is
    ``(||r|| > atol + rtol*||r(u0)||) & (k < max_steps)``. Leaving on the *second* clause returns the
    last iterate with no signal whatsoever, so a stalled solve is indistinguishable downstream from a
    converged one -- it comes back as a perfectly plausible-looking field. The steady-linear default
    already refuses to do that (``_fem._residual_check`` / ``solver_api._maybe_residual_check``); the
    nonlinear path had no equivalent, which is the gap this closes.

    No-op under ``jit``/``vmap``/``grad``: the test needs a concrete residual, so it would both force a
    device->host sync and fail to concretise. Same guard, and same trade, as the two linear checks --
    under a transform the solver's own iteration cap is all there is.
    """
    if any(isinstance(v, jax.core.Tracer) for v in (u, u0)):
        return u
    rn = float(jnp.linalg.norm(f0(u)))
    bound = atol + rtol * float(jnp.linalg.norm(f0(u0)))
    if not math.isfinite(rn) or rn > bound:
        raise RuntimeError(
            f"{who} did not converge in max_steps={max_steps}: residual norm {rn:.3e} against the "
            f"tolerance atol + rtol*||r(u0)|| = {bound:.3e} (atol={atol:g}, rtol={rtol:g}). The last "
            "iterate is NOT a root -- raise max_steps, loosen atol/rtol, globalize the iteration "
            "(jno.solve.newton(line_search=True) or damping<1), or start from a better x0."
        )
    return u


def bicgstab(matvec, b, *, tol=1e-10, maxit=2000):
    """Matrix-free BiCGStab; returns ``x`` solving ``matvec(x) = b`` (general matrices)."""
    bnorm = jnp.linalg.norm(b)

    def cond(s):
        _, r, *_, k = s
        return (jnp.linalg.norm(r) > tol * bnorm) & (k < maxit)

    def body(s):
        x, r, rhat, rho, alpha, omega, v, p, k = s
        rho_new = rhat @ r
        beta = (rho_new / (rho + _EPS)) * (alpha / (omega + _EPS))
        p = r + beta * (p - omega * v)
        v = matvec(p)
        alpha = rho_new / (rhat @ v + _EPS)
        sv = r - alpha * v
        t = matvec(sv)
        omega = (t @ sv) / (t @ t + _EPS)
        x = x + alpha * p + omega * sv
        r = sv - omega * t
        return x, r, rhat, rho_new, alpha, omega, v, p, k + 1

    z = jnp.zeros_like(b)
    one = jnp.array(1.0, b.dtype)
    x0 = jnp.zeros_like(b)
    state = (x0, b - matvec(x0), b, one, one, one, z, z, 0)
    x, *_ = jax.lax.while_loop(cond, body, state)
    return x


def _linsolve(matvec, b, *, tol, maxit):
    """Transposable black-box linear solve: BiCGStab forward, BiCGStab on ``J^T`` for the
    reverse rule (via ``custom_linear_solve``) -- so it never differentiates the loop."""
    solve = lambda mv, rhs: bicgstab(mv, rhs, tol=tol, maxit=maxit)
    return jax.lax.custom_linear_solve(matvec, b, solve, transpose_solve=solve)


def newton_krylov(
    residual_fn,
    u0,
    *,
    rtol=1e-8,
    atol=1e-8,
    max_steps=100,
    inner_tol=1e-10,
    inner_maxit=2000,
    linear_solve=None,
    damping=1.0,
    line_search=False,
    ls_max=25,
    ls_c=1e-4,
):
    """Root-find ``residual_fn(u) = 0`` from guess ``u0``; differentiable w.r.t. any value
    ``residual_fn`` closes over. Drop-in for the ``(residual_fn, u0) -> u`` solver contract.

    ``linear_solve`` overrides the inner matrix-free solve: a ``(matvec, rhs) -> x`` callable
    (e.g. adapted from a ``jno.solve`` Krylov solver); it serves both the Newton step and the
    implicit-diff tangent solve. Default: BiCGStab wrapped in ``custom_linear_solve``.

    ``damping`` scales each update ``u += damping * delta`` (0 < damping <= 1): a fixed
    relaxation that trades steps for robustness on strongly nonlinear residuals. With
    ``jno.lag``-frozen coefficients in the residual the linearization is the *Picard* operator
    and this loop is the damped Picard iteration (see :func:`jno.solve.picard`).

    ``line_search`` enables adaptive backtracking (a residual-norm Armijo globalization) on top of
    the fixed ``damping``: each step tries ``alpha = damping`` and halves it (up to ``ls_max``
    times) until ``||f(u + alpha*delta)|| <= (1 - ls_c*alpha)*||f(u)||``. Fixed damping alone can
    overshoot and diverge on stiff residuals (e.g. a rigid-plastic cold start where the effective
    viscosity spans several orders of magnitude); backtracking finds a safe step automatically, so
    the same driver converges without hand-tuning ``damping``. Off by default (behaviour unchanged).
    The line search is a bounded ``jax.lax.while_loop`` -- differentiability is untouched, since the
    implicit-diff tangent solve depends only on the converged root, not the path taken to it."""
    # residual functions can hand back a plain numpy array for concrete inputs; coerce so the
    # jax.lax.custom_root primitive only ever sees JAX values.
    f0 = lambda u: jnp.asarray(residual_fn(u)).reshape(-1)
    u0 = jnp.asarray(u0).reshape(-1)
    # The inner solve serves BOTH the Newton step AND ``custom_root``'s implicit-diff
    # tangent/adjoint solve (``tangent_solve`` below), so it must be reverse-transposable. The
    # default ``_linsolve`` already wraps BiCGStab in ``custom_linear_solve`` with a transpose rule.
    # A slot solver (e.g. ``jno.solve.gmres``) is a *raw* Krylov call with no transpose rule of its
    # own; used as ``tangent_solve`` unwrapped, it differentiates in isolation (steady) but breaks when
    # such solves are *chained* -- e.g. a transient time-march of Newton steps -- because JAX then tries
    # to transpose its ``custom_linear_solve`` w.r.t. the operator and raises NotImplementedError. So we
    # firewall the slot the same way: wrap it in ``custom_linear_solve`` with an explicit transpose
    # solve that runs the *same* solver on ``A^T`` (custom_linear_solve hands ``transpose_solve`` the
    # transposed matvec). This makes the adjoint well-defined for every inner solver, not just the
    # default -- the fix is at the single point where transposability is actually required.
    if linear_solve is None:
        inner = lambda mv, rhs: _linsolve(mv, rhs, tol=inner_tol, maxit=inner_maxit)
    else:
        _slot = linear_solve
        inner = lambda mv, rhs: jax.lax.custom_linear_solve(mv, rhs, _slot, transpose_solve=_slot)

    def _backtrack(f, u, delta, rn):
        """First ``alpha`` in ``damping * 0.5^i`` meeting residual-norm Armijo; else the last (tiny)."""

        def cond(s):
            _a, accepted, it = s
            return (~accepted) & (it < ls_max)

        def step(s):
            a, _acc, it = s
            accepted = jnp.linalg.norm(f(u + a * delta)) <= (1.0 - ls_c * a) * rn
            return jnp.where(accepted, a, 0.5 * a), accepted, it + 1

        a, _acc, _it = jax.lax.while_loop(cond, step, (jnp.asarray(damping, u.dtype), False, 0))
        return a

    def solve(f, x0):
        r0n = jnp.linalg.norm(f(x0))

        def cond(state):
            _, r, k = state
            return (jnp.linalg.norm(r) > atol + rtol * r0n) & (k < max_steps)

        def body(state):
            u, _r, k = state
            ru, jvp = jax.linearize(f, u)  # ru = f(u); jvp(v) = J @ v, reused across inner iters
            delta = inner(jvp, -ru)
            alpha = _backtrack(f, u, delta, jnp.linalg.norm(ru)) if line_search else damping
            u = u + alpha * delta
            return u, f(u), k + 1

        u, _r, _k = jax.lax.while_loop(cond, body, (x0, f(x0), 0))
        return u

    tangent_solve = lambda g, y: inner(g, y)
    root = jax.lax.custom_root(f0, u0, solve, tangent_solve)
    # Checked OUTSIDE custom_root: everything inside `solve` is traced, so an in-loop guard could
    # never concretise. Here `root` is concrete whenever the caller was eager, which is exactly when
    # the check can do any good.
    return _convergence_check(f0, u0, root, rtol=rtol, atol=atol, max_steps=max_steps, who="newton_krylov")


def newton_direct(
    residual_fn,
    jacobian_fn,
    u0,
    *,
    rtol=1e-8,
    atol=1e-8,
    max_steps=100,
    damping=1.0,
    line_search=False,
    ls_max=25,
    ls_c=1e-4,
    linear_solve=None,
):
    """Root-find ``residual_fn(u) = 0`` with a **sparse-direct** Newton: each step solves against the
    ASSEMBLED Jacobian ``jacobian_fn(u)`` (a ``jax.experimental.sparse.BCOO``) instead of the
    matrix-free Krylov inner solve of :func:`newton_krylov`. A direct factorization is
    robust on **indefinite / ill-conditioned** systems -- Taylor-Hood velocity/pressure saddles, stiff
    phase-change (Carman-Kozeny) drag -- where BiCGStab stalls (no saddle-point preconditioner).

    ``linear_solve`` is an ``(A, b) -> x`` callable over the ASSEMBLED tangent -- the composed
    ``linear=``/``precond=`` slots. Default: ``sparse_lu_solve``. Without it the driver hardcoded
    cuSolver, so ``fem.solve(nonlinear=newton(direct=True), linear=lu(backend="host"))`` silently
    ignored the placement it was asked for -- and this is the one path where that choice decides
    whether the problem runs at all, cuSolver being the ceiling on exactly the saddle systems the
    direct Newton exists for.

    **This is the path ``lu(backend="cudss")`` was worth adding for.** A Newton step changes the
    tangent's VALUES and holds its SPARSITY fixed, which is exactly the split cuDSS exposes and the
    one host SuperLU cannot: the symbolic plan is computed once and every subsequent step pays only
    the numeric factorization (measured 64.7x per step against ``backend="host"`` at n=64,000).

    Differentiable w.r.t. anything ``residual_fn`` closes over: the forward Newton runs to the root
    un-differentiated, then ``jax.lax.custom_root`` provides the implicit-function-theorem gradient via a
    **direct, transposable** tangent solve on the Jacobian assembled at the root (so the reverse pass
    solves ``Jᵀ`` directly too, not with a stalling Krylov). That transpose is the one requirement on a
    supplied ``linear_solve``: it is called on ``Jᵀ`` as well as ``J``. ``jacobian_fn(u)`` must return the
    assembled Jacobian of ``residual_fn`` at ``u``. ``damping`` / ``line_search`` as in
    :func:`newton_krylov`."""
    if linear_solve is None:
        from .linear import sparse_lu_solve

        linear_solve = sparse_lu_solve

    f0 = lambda u: jnp.asarray(residual_fn(u)).reshape(-1)  # noqa: E731
    u0 = jnp.asarray(u0).reshape(-1)

    def _backtrack(u, delta, rn):  # residual-norm Armijo, identical to newton_krylov's
        def cond(s):
            _a, acc, it = s
            return (~acc) & (it < ls_max)

        def step(s):
            a, _acc, it = s
            acc = jnp.linalg.norm(f0(u + a * delta)) <= (1.0 - ls_c * a) * rn
            return jnp.where(acc, a, 0.5 * a), acc, it + 1

        a, _acc, _it = jax.lax.while_loop(cond, step, (jnp.asarray(damping, u.dtype), False, 0))
        return a

    def _forward(x0):
        r0n = jnp.linalg.norm(f0(x0))

        def cond(state):
            _u, r, k = state
            return (jnp.linalg.norm(r) > atol + rtol * r0n) & (k < max_steps)

        def body(state):
            u, _r, k = state
            r = f0(u)
            delta = linear_solve(jacobian_fn(u), -r)  # DIRECT solve of the assembled tangent
            alpha = _backtrack(u, delta, jnp.linalg.norm(r)) if line_search else damping
            u = u + alpha * delta
            return u, f0(u), k + 1

        u, _r, _k = jax.lax.while_loop(cond, body, (x0, f0(x0), 0))
        return u

    root = _forward(u0)  # un-differentiated forward solve; custom_root below supplies the gradient
    _convergence_check(f0, u0, root, rtol=rtol, atol=atol, max_steps=max_steps, who="newton_direct")

    def _tangent(g, y):  # solve J_root x = y (and Jᵀ on the reverse pass) DIRECTLY at the converged root
        J = jacobian_fn(root)
        fwd = lambda _mv, rhs: linear_solve(J, rhs)  # noqa: E731
        tsp = lambda _mv, rhs: linear_solve(J.T, rhs)  # noqa: E731
        return jax.lax.custom_linear_solve(g, y, fwd, transpose_solve=tsp)

    return jax.lax.custom_root(f0, root, lambda _f, _x0: root, _tangent)


def staggered_newton(
    residual_fn,
    u0,
    blocks,
    *,
    rtol=1e-8,
    atol=1e-8,
    max_sweeps=200,
    inner_steps=20,
    inner_tol=1e-10,
    inner_maxit=2000,
    linear_solve=None,
    damping=1.0,
    line_search=False,
    ls_max=25,
    ls_c=1e-4,
):
    """**Alternate minimization** (staggered / operator-split) root find over a block-partitioned system.

    Sweep the blocks in order, solving each field's equations with the others held fixed, and repeat
    until the FULL residual is converged. Gauss-Seidel in the blocks: each sub-solve sees the updates
    made earlier in the same sweep.

    Why it exists: a coupled energy can be **non-convex in the fields jointly while convex in each
    separately**, and a monolithic Newton then has no descent guarantee and diverges. The canonical case
    is variational phase-field fracture, where the ``(1-d)^2 |grad u|^2`` coupling is quartic in the pair
    but each field's own problem is a linear elliptic one. Alternate minimization turns that into a
    sequence of convex solves, each decreasing the energy. Introduced for exactly this by Bourdin,
    Francfort & Marigo, *Numerical experiments in revisited brittle fracture*, J. Mech. Phys. Solids
    **48** (2000), §3; as the staggered operator split with a history field by Miehe, Welschinger &
    Hofacker, IJNME **83** (2010). The trade is convergence RATE: alternate minimization is linear where
    Newton is quadratic, hence the high ``max_sweeps`` default (see Farrell & Maurini, *Linear and
    nonlinear solvers for variational phase-field models of brittle fracture*, CMAME **312**, 2017).

    ``blocks`` is a list of index arrays, one per field, partitioning the DOF vector. Each sub-solve is
    an ordinary matrix-free Newton on the restricted residual ``x -> R(u with block set to x)[block]``,
    so ``jax.linearize`` gives that field's DIAGONAL Jacobian block, matrix-free, with no assembly.

    Differentiable, and for the ordinary reason: at convergence the full residual is zero, so the sweep
    is just *a way of finding that root*. ``lax.custom_root`` therefore wraps the whole loop with the
    tangent solve on the FULL Jacobian — the implicit-function theorem does not care how the root was
    reached, and the alternating structure disappears from the derivative. (Which also means the tangent
    solve is a coupled linear solve even though the forward iteration never formed one; that is correct,
    and it is cheap, because the tangent at a converged root is well behaved even when the nonlinear
    monolithic iteration was not.)
    """
    f0 = lambda u: jnp.asarray(residual_fn(u)).reshape(-1)  # noqa: E731
    u0 = jnp.asarray(u0).reshape(-1)
    idx = [jnp.asarray(b, dtype=jnp.int32) for b in blocks]
    if linear_solve is None:
        inner = lambda mv, rhs: _linsolve(mv, rhs, tol=inner_tol, maxit=inner_maxit)  # noqa: E731
    else:
        _slot = linear_solve
        inner = lambda mv, rhs: jax.lax.custom_linear_solve(mv, rhs, _slot, transpose_solve=_slot)  # noqa: E731

    def _sweep(u):
        """One Gauss-Seidel pass: solve each block's own equations with the others frozen."""
        for b in idx:

            def _restricted(x, _u=u, _b=b):
                return f0(_u.at[_b].set(x))[_b]

            # No `custom_root` on the sub-solve: the OUTER one below supplies every derivative, so the
            # inner iteration only has to land on the sub-root.
            u = u.at[b].set(_newton_inner(_restricted, u[b]))
        return u

    def _newton_inner(f, x0):
        def cond(state):
            _x, r, k = state
            return (jnp.linalg.norm(r) > atol) & (k < inner_steps)

        def body(state):
            x, _r, k = state
            rx, jvp = jax.linearize(f, x)
            dx = inner(jvp, -rx)
            a = _inner_backtrack(f, x, dx, jnp.linalg.norm(rx)) if line_search else damping
            x = x + a * dx
            return x, f(x), k + 1

        x, _r, _k = jax.lax.while_loop(cond, body, (x0, f(x0), 0))
        return x

    def _inner_backtrack(f, x, dx, rn):
        def cond(s):
            _a, ok, it = s
            return (~ok) & (it < ls_max)

        def step(s):
            a, _ok, it = s
            ok = jnp.linalg.norm(f(x + a * dx)) <= (1.0 - ls_c * a) * rn
            return jnp.where(ok, a, 0.5 * a), ok, it + 1

        a, _ok, _it = jax.lax.while_loop(cond, step, (jnp.asarray(damping, x.dtype), False, 0))
        return a

    def solve(f, x0):
        r0n = jnp.linalg.norm(f(x0))

        def cond(state):
            _u, r, k = state
            return (jnp.linalg.norm(r) > atol + rtol * r0n) & (k < max_sweeps)

        def body(state):
            u, _r, k = state
            u = _sweep(u)
            return u, f(u), k + 1

        u, _r, _k = jax.lax.while_loop(cond, body, (x0, f(x0), 0))
        return u

    tangent_solve = lambda g, y: inner(g, y)  # noqa: E731
    root = jax.lax.custom_root(f0, u0, solve, tangent_solve)
    return _convergence_check(f0, u0, root, rtol=rtol, atol=atol, max_steps=max_sweeps, who="staggered")
