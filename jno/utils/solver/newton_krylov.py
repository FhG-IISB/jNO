"""The nonlinear root-finders -- the optimistix-free drivers behind ``fem.solve(nonlinear=...)``.

Three of them, sharing one convergence check and one step-retreat rule: matrix-free Newton-Krylov (the
default, below), sparse-direct Newton, and alternate minimization. Only the first is Jacobian-free; the
other two take the assembled tangent, for reasons their own docstrings give.

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
import numpy as np

__all__ = ["newton_krylov", "newton_direct", "staggered_newton", "bicgstab"]

_EPS = 1e-300


#: Last eager nonlinear solve's outcome, written by ``_convergence_check`` and read by
#: ``fem.solve`` into ``fem.stats``. A module-level slot rather than a return-value change on
#: purpose: the drivers' ``(residual_fn, u0) -> u`` contract has many callers, and observability
#: must not alter it. Under jit/vmap/grad the check self-disables (concrete-only, as documented
#: below) and this slot simply keeps its previous content -- the same silence the guard itself has.
LAST_NEWTON_STATS: dict = {}


def _convergence_check(f0, u0, u, *, rtol, atol, max_steps, who, steps=None):
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
    LAST_NEWTON_STATS.clear()
    LAST_NEWTON_STATS.update(
        driver=who,
        residual=rn,
        bound=bound,
        steps=None if steps is None or isinstance(steps, jax.core.Tracer) else int(steps),
        converged=bool(math.isfinite(rn) and rn <= bound),
    )
    if not math.isfinite(rn) or rn > bound:
        raise RuntimeError(
            f"{who} did not converge in max_steps={max_steps}: residual norm {rn:.3e} against the "
            f"tolerance atol + rtol*||r(u0)|| = {bound:.3e} (atol={atol:g}, rtol={rtol:g}). The last "
            "iterate is NOT a root -- raise max_steps, loosen atol/rtol, globalize the iteration "
            "(jno.solve.newton(line_search=True) or damping<1), or start from a better x0."
        )
    return u


def _retreat(accept, *, hi, lo, max_halvings, dtype):
    """The largest parameter in ``[lo, hi]`` that ``accept`` likes, found by bisecting toward ``lo``.

    **The one place this library shortens a step.** Every driver that moves the iterate past a point
    some sub-problem actually solved for retreats through here, so the rule is stated once:

        any step that goes beyond what was solved for needs an admissibility test *and* a known-good
        point to fall back to.

    The instantiations differ only in where they retreat TO and what "admissible" means:

    ================================  ====================================  =================================
    step                              ``lo`` (the fallback)                 ``accept``
    ================================  ====================================  =================================
    Newton (root-find)                ``0`` -- do not move                  :func:`_armijo` on ``||R||``
    extrapolation (over-relaxation)   ``1`` -- the sub-solve's own answer   finite residual, and feasible
    ================================  ====================================  =================================

    Retreating a Newton step toward 0 is standard. Retreating an *extrapolation* toward 1 is the
    interesting one: ``1`` means "just take the sub-solve's answer", which was already evaluated on the
    way here and is therefore admissible by construction -- so an over-relaxed iteration can never do
    worse than the un-relaxed one. Farrell & Maurini's Algorithm 2 already bisects ``[1, omega]`` this
    way for the bound constraint (IJNME **109** (2017), section 2.1); the same rule carries finiteness.

    ``accept`` returning a NaN-poisoned comparison is *correct behaviour*: ``nan <= x`` is False, so a
    trial point where the residual is not finite is rejected and the step shortens. This is the same
    NaN-safe-by-construction idiom the mesh relocator uses (``fem_adapt`` writes ``not (x > 0)`` rather
    than ``x <= 0`` for exactly this reason).

    Exhausting ``max_halvings`` returns the last (smallest) trial rather than ``lo``, which is the
    historic behaviour of the three line searches this replaces: the caller still takes a step, and the
    outer convergence check is what refuses a solve that never got anywhere.
    """
    # `0.5 * a` verbatim when lo == 0 -- the Armijo path must stay bit-identical to the three
    # hand-written loops this replaces, and `lo + 0.5*(a-lo)` is only *mathematically* the same.
    _next = (lambda a: 0.5 * a) if lo == 0.0 else (lambda a: lo + 0.5 * (a - lo))

    def cond(s):
        _a, ok, it = s
        return (~ok) & (it < max_halvings)

    def body(s):
        a, _ok, it = s
        ok = accept(a)
        return jnp.where(ok, a, _next(a)), ok, it + 1

    a, _ok, _it = jax.lax.while_loop(cond, body, (jnp.asarray(hi, dtype), False, 0))
    return a


def _armijo(f, x, delta, rn, *, ls_c):
    """The residual-norm Armijo predicate shared by every Newton-type step: ``accept(a)``."""
    return lambda a: jnp.linalg.norm(f(x + a * delta)) <= (1.0 - ls_c * a) * rn


def _bisect_slope(f, x, delta, *, atol, rtol, max_iters, dtype):
    """**Exact** line search: the ``lam`` that MINIMIZES the energy along ``delta``, by bisection.

    The other line search here, :func:`_retreat` with :func:`_armijo`, is *backtracking* -- it accepts the
    first step length that is good enough. This one solves the one-dimensional problem instead. With
    ``phi(lam) = E(x + lam*delta)``, its derivative is

        ``phi'(lam) = R(x + lam*delta) . delta``

    which is computable from the RESIDUAL alone -- no energy value, no second derivatives. That is what
    makes an exact line search possible at all in jNO, whose weak-form contract carries no energy
    functional. For a strictly convex subproblem ``phi'`` is monotonically increasing, so it has a unique
    root and that root is the minimizer.

        ``phi'(0)*phi'(1) > 0``  -- no sign change, no interior minimum: take ``lam = 1``
        otherwise                -- bisect ``[0, 1]`` on the sign of ``phi'``

    Algorithm: Heinzmann, Vicentini, Carrara et al., *Iterative convergence in phase-field brittle
    fracture computations: exact line search is all you need*, Computational Mechanics (2026),
    arXiv:2511.23064, **section 3, Algorithm 2**; contributed to PETSc as ``SNESLineSearchBisection``.
    Their Proposition 1 gives convergence of the bisection, Proposition 2 global convergence of Newton
    with it, and Remark 4 chains those into convergence of the whole alternate-minimization scheme --
    provided each subproblem is strictly convex and coercive.

    **Why not backtracking.** Because it demonstrably stalls on this problem class, from both directions.
    That paper measures backtracking as the WORST of the line searches it compares ("significantly longer
    staggered paths ... the algorithm can exit with non-optimal step size multipliers"), and its
    energy-based variant reached convergence in only one of their test cases. Measured here independently
    on a phase-field energy: at a step whose direction was strongly descending (``phi'(0) = -1.18``) the
    residual norm ROSE from 0.36 to 0.90, so the residual-norm Armijo rejected every one of its 25
    halvings and the iterate froze permanently. The slope at the full step was ``+0.40`` -- a minimum sat
    inside the step, bracketed, waiting to be found by exactly this.

    Cost is extra residual evaluations per Newton step, which their paper notes matters less as the
    linear solve grows. In jNO it matters even less: a residual is ~14.6 ms against a ~613 ms tangent
    assembly on the problem this was profiled on.

    NaN-safe by the same construction as everything else in this module: a non-finite ``phi'`` fails its
    sign comparison, so the bracket shrinks rather than the NaN propagating."""
    slope = lambda lam: jnp.dot(jnp.asarray(f(x + lam * delta)).reshape(-1), delta)  # noqa: E731
    s0, s1 = slope(jnp.zeros((), dtype)), slope(jnp.ones((), dtype))
    # Scaled by ||delta|| so `atol` is a slope in physical units rather than a raw dot product.
    tol = atol * jnp.linalg.norm(delta) + rtol * jnp.abs(s0)

    def cond(st):
        lo, hi, sl, it = st
        return (jnp.abs(sl) > tol) & ((hi - lo) > 1e-12) & (it < max_iters)

    def body(st):
        lo, hi, _sl, it = st
        mid = 0.5 * (lo + hi)
        sm = slope(mid)
        # `sm * s0 > 0` -> the root is to the RIGHT of mid (same sign as the left end), else to the left.
        right = sm * s0 > 0.0
        return jnp.where(right, mid, lo), jnp.where(right, hi, mid), sm, it + 1

    # The carried slope starts at INFINITY, not at ``s0``. Seeding it with ``s0`` lets the loop exit
    # before bisecting even once whenever ``|phi'(0)|`` is already under tolerance -- and it then returns
    # the midpoint of an UNREFINED bracket, ``lam = 0.5``. That is not a small error: a small
    # ``phi'(0)`` is the signature of being near convergence, so the effect is to halve the final Newton
    # step of every sub-solve. Measured before the fix: an exact line search needing MORE staggered
    # sweeps than the backtracking it replaced (22 vs 21), and a probe whose minimum lay at ``lam = 0``
    # returning ``lam = 0.5`` with ``phi' = +1.0``. The convergence test belongs on midpoints only.
    lo, hi, _sl, _it = jax.lax.while_loop(
        cond, body, (jnp.zeros((), dtype), jnp.ones((), dtype), jnp.asarray(jnp.inf, dtype), 0)
    )
    lam = 0.5 * (lo + hi)
    # No sign change over [0, 1]: phi' never crosses zero, so the energy is still decreasing at the full
    # step and there is nothing to find inside it (their Fig. 2a).
    return jnp.where(s0 * s1 > 0.0, jnp.ones((), dtype), lam)


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
        return _retreat(_armijo(f, u, delta, rn, ls_c=ls_c), hi=damping, lo=0.0, max_halvings=ls_max, dtype=u.dtype)

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

    def _backtrack(u, delta, rn):  # residual-norm Armijo -- the same retreat newton_krylov uses
        return _retreat(_armijo(f0, u, delta, rn, ls_c=ls_c), hi=damping, lo=0.0, max_halvings=ls_max, dtype=u.dtype)

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

        u, _r, k = jax.lax.while_loop(cond, body, (x0, f0(x0), 0))
        return u, k

    root, _steps = _forward(u0)  # un-differentiated forward solve; custom_root supplies the gradient
    _convergence_check(f0, u0, root, rtol=rtol, atol=atol, max_steps=max_steps, who="newton_direct", steps=_steps)

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
    jacobian=None,
    over_relax=1.0,
    project=None,
    constrained=None,
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

    ``jacobian`` (``u -> BCOO``, from ``jno.solve.staggered(direct=True)``) switches those sub-solves to
    a **sparse-direct** factorization instead. It exists because the matrix-free default has no
    preconditioner to offer a sub-block: ``precond=`` materializes against an assembled operator, and a
    restriction closure has none, so a near-incompressible displacement block is solved by *unpreconditioned*
    BiCGStab -- which is where alternate minimization's time actually goes.

    The extraction is the trick that keeps this simple. Rather than slice ``J[b][:, b]`` out (a
    data-dependent nnz, so not a static shape), zero the COMPLEMENT's rows and columns and put a unit
    diagonal there::

        J_b = [[J_bb, 0  ],      solved against  rhs = [-r_b, 0]
               [0,    I  ]]

    which is the same n x n system with the other fields frozen -- ``x`` is the sub-block solve on ``b``
    and exactly zero off it. The padding is pure diagonal, so it adds no fill-in and the factorization
    cost stays the block's. It also reuses the elimination helpers already in ``fem_utils``.

    The cost is honest and worth stating: the FULL tangent is assembled to use one block of it. Against
    an unpreconditioned Krylov on an ill-conditioned block that is still the better trade, and a
    sparsity-caching backend (``lu(backend="pardiso"/"cudss")``) pays only the numeric re-factorization
    per sweep -- but on a well-conditioned problem the matrix-free default remains cheaper.

    ``over_relax`` (omega) attacks the OTHER cost: the number of sweeps. Alternate minimization is
    exactly a nonlinear block Gauss-Seidel iteration, and over-relaxation accelerates that the same way
    it accelerates the linear one -- take each sub-step's update direction and go ``omega`` times as far:

        x_new = x_old + omega * (subsolve(x_old) - x_old)      applied PER BLOCK, not per sweep

    Farrell & Maurini, *Linear and nonlinear solvers for variational phase-field models of brittle
    fracture*, IJNME **109** (2017) 648-667, **Algorithm 2 (ORAM), section 2.1**. ``omega = 1`` is
    ordinary alternate minimization, bit-for-bit. Kahan's classical bound gives ``omega in (0, 2)`` as
    necessary for SOR to converge; the paper reports convergence for every ``omega`` in that range it
    tried, and no way to pick the best one a priori -- their words, "we rely on the naive strategy of
    numerical experimentation on coarser problems". So this is a knob with no default worth guessing,
    which is why it defaults to 1.

    ``project`` keeps a BOX-CONSTRAINED field feasible after that extrapolation: the sub-solve's own
    answer lies in the box, but stepping *past* it need not. Deviation worth stating -- the paper backs
    the scalar off (largest ``omega_bar`` in ``(0, omega)`` that stays feasible, section 2.1); this
    clips componentwise instead, which is also feasible and keeps more of the step, but it is not their
    rule. Without a projector the extrapolated iterate may leave the box; that is harmless for the
    ANSWER (the min-map's root is feasible by construction) but it is a different iteration.

    ``constrained`` is the dof set carrying an ESSENTIAL condition, and over-relaxation must skip it.
    The paper's ``u~`` lives in the constrained space ``C_u``, where a prescribed dof has ``delta = 0``
    so ``omega`` cannot reach it; jNO imposes the condition as a residual ROW, so the sub-solve lands
    exactly on the prescribed value and extrapolating past it is simply wrong. Measured on one row with
    ``g = 2``: ``omega = 1.7`` gave 3.40 -> 1.02 -> 2.69, an oscillation decaying only as
    ``|1-omega|^k``, with every other field solved against that wrong boundary value throughout. Worst
    on a *ramped* condition (a load-path march), where ``g`` moves every step so the overshoot recurs.

    **The extrapolation is guarded** -- see :func:`_retreat`. It bisects ``[1, omega]`` until the trial
    point has a finite residual and is feasible, and ``omega = 1`` is the sub-solve's own answer, so the
    worst case is a degrade to plain alternate minimization rather than a divergence. That is what makes
    it usable on a finite-strain form at all: unguarded, stepping past a converged answer inverts an
    element (``det F <= 0``, so ``J**(-2/3)`` is NaN) and the 3-D Yeoh SENT march died on load step 1 at
    every ``omega`` down to 1.1.

    The predicate is finiteness and feasibility, NOT descent -- over-relaxation is deliberately not a
    descent method, so a residual-decrease test would reject nearly every ``omega > 1``.

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
    n_dofs = int(u0.shape[0])
    # Per block, the mask of dofs over-relaxation may move: everything except the ones carrying an
    # essential condition. `None` when the problem declares none, so the ordinary path allocates nothing.
    if over_relax == 1.0 or constrained is None or len(np.asarray(constrained)) == 0:
        free = [None] * len(blocks)
    else:
        _pinned = np.zeros(n_dofs, dtype=bool)
        _pinned[np.asarray(constrained, dtype=int)] = True
        free = [jnp.asarray(~_pinned[np.asarray(b, dtype=int)]) for b in blocks]
    direct = jacobian is not None
    if direct:
        from .fem_utils import bcoo_set_unit_diag, bcoo_zero_rows_cols

        if linear_solve is None:
            from .linear import sparse_lu_solve

            _dsolve = sparse_lu_solve
        else:
            _dsolve = linear_solve
        # The complement of each block, computed EAGERLY -- the partition is static (it comes from the
        # problem's DOF layout), so this is a host-side set operation, not a traced one.
        _all = np.arange(n_dofs)
        comps = [jnp.asarray(np.setdiff1d(_all, np.asarray(b)), dtype=jnp.int32) for b in blocks]
        inner = None
    elif linear_solve is None:
        inner = lambda mv, rhs: _linsolve(mv, rhs, tol=inner_tol, maxit=inner_maxit)  # noqa: E731
    else:
        _slot = linear_solve
        inner = lambda mv, rhs: jax.lax.custom_linear_solve(mv, rhs, _slot, transpose_solve=_slot)  # noqa: E731

    def _block_step(u_full, b, comp):
        """The sparse-direct Newton step for one block: freeze the complement to the identity, solve."""

        def step(x, rx):
            J = jacobian(u_full.at[b].set(x))
            J_b = bcoo_set_unit_diag(bcoo_zero_rows_cols(J, comp), comp)
            rhs = jnp.zeros(n_dofs, rx.dtype).at[b].set(-rx)
            return _dsolve(J_b, rhs)[b]

        return step

    def _sweep(u):
        """One Gauss-Seidel pass: solve each block's own equations with the others frozen."""
        for j, b in enumerate(idx):

            def _restricted(x, _u=u, _b=b):
                return f0(_u.at[_b].set(x))[_b]

            # No `custom_root` on the sub-solve: the OUTER one below supplies every derivative, so the
            # inner iteration only has to land on the sub-root.
            step = _block_step(u, b, comps[j]) if direct else None
            x_sub = _newton_inner(_restricted, u[b], step)
            if over_relax == 1.0:
                u = u.at[b].set(x_sub)
            else:
                # ORAM: extrapolate along this sub-step's own update direction (Farrell & Maurini,
                # Algorithm 2) -- PER BLOCK, so the next block already sees the over-relaxed value,
                # which is what keeps it a Gauss-Seidel iteration rather than a Jacobi one.
                def _at(w, _u=u, _b=b, _x=x_sub, _f=free[j]):
                    # `where(w == 1, x_sub, ...)`: the fallback must be EXACTLY the sub-solve's answer,
                    # and `a + 1.0*(b - a)` is not exactly `b` in floating point.
                    xr = jnp.where(w == 1.0, _x, _u[_b] + w * (_x - _u[_b]))
                    # ...and only on the FREE dofs. The paper's u~ lives in the constrained space C_u,
                    # so a prescribed dof has delta = 0 there and omega never reaches it. jNO imposes
                    # essential conditions as residual rows, so the sub-solve lands exactly ON the
                    # prescribed value and extrapolating past it is simply wrong: measured on one row
                    # with g = 2, omega = 1.7 gave 3.40 -> 1.02 -> 2.69, decaying only as |1-omega|^k.
                    return _u.at[_b].set(xr if _f is None else jnp.where(_f, xr, _x))

                def _admissible(w):
                    """Finite, and feasible. NOT a descent test -- over-relaxation is deliberately not a
                    descent method, so demanding a residual decrease would reject almost every w > 1 and
                    silently collapse the feature back to plain alternate minimization."""
                    trial = _at(w)
                    ok = jnp.all(jnp.isfinite(f0(trial)))
                    return ok if project is None else ok & jnp.all(project(trial) == trial)

                # Retreat toward w = 1, which IS the sub-solve's own answer -- already evaluated on the
                # way here, hence admissible by construction. So an over-relaxed sweep can never do worse
                # than the un-relaxed one; the failure this fixes was an unguarded step into an inverted
                # element (det F <= 0 -> J**(-2/3) = NaN) on the very first load step of a Yeoh march.
                u = _at(_retreat(_admissible, hi=over_relax, lo=1.0, max_halvings=ls_max, dtype=u.dtype))
            if project is not None:
                u = project(u)  # safety net; the retreat above already chose a feasible w
        return u

    def _newton_inner(f, x0, step=None):
        def cond(state):
            _x, r, k = state
            return (jnp.linalg.norm(r) > atol) & (k < inner_steps)

        def body(state):
            x, _r, k = state
            if step is None:
                rx, jvp = jax.linearize(f, x)
                dx = inner(jvp, -rx)
            else:
                rx = f(x)
                dx = step(x, rx)
            a = _inner_step_length(f, x, dx, jnp.linalg.norm(rx))
            x = x + a * dx
            return x, f(x), k + 1

        x, _r, _k = jax.lax.while_loop(cond, body, (x0, f(x0), 0))
        return x

    def _inner_step_length(f, x, dx, rn):
        """How far to move along the sub-problem's Newton direction.

        ``line_search=True`` (the default) is the EXACT line search: bisect for the root of the
        directional derivative, i.e. the minimizer of the energy along ``dx``. That is the step this
        driver's convergence guarantee rests on -- see :func:`_bisect_slope`. ``"backtrack"`` keeps the
        older residual-norm Armijo, which is what this used to do and which the literature (and a
        measurement here) rank last. ``False`` takes the fixed ``damping``."""
        if line_search is False:
            return damping
        if line_search == "backtrack":
            return _retreat(_armijo(f, x, dx, rn, ls_c=ls_c), hi=damping, lo=0.0, max_halvings=ls_max, dtype=x.dtype)
        return damping * _bisect_slope(f, x, dx, atol=atol, rtol=ls_c, max_iters=ls_max, dtype=x.dtype)

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

    if direct:
        # Same shape as `newton_direct`: run the sweep undifferentiated, then hang `custom_root` off the
        # root it found so the tangent (and its transpose, for reverse mode) is a DIRECT solve on the
        # FULL assembled Jacobian. The alternating structure is absent from the derivative either way --
        # the implicit-function theorem does not care how the root was reached — but a direct tangent is
        # the consistent choice here: the caller picked a direct slot precisely because Krylov stalls.
        root_val = solve(f0, u0)

        def _tangent(g, y):
            J = jacobian(root_val)
            fwd = lambda _mv, rhs: _dsolve(J, rhs)  # noqa: E731
            tsp = lambda _mv, rhs: _dsolve(J.T, rhs)  # noqa: E731
            return jax.lax.custom_linear_solve(g, y, fwd, transpose_solve=tsp)

        root = jax.lax.custom_root(f0, root_val, lambda _f, _x0: root_val, _tangent)
    else:
        tangent_solve = lambda g, y: inner(g, y)  # noqa: E731
        root = jax.lax.custom_root(f0, u0, solve, tangent_solve)
    return _convergence_check(f0, u0, root, rtol=rtol, atol=atol, max_steps=max_sweeps, who="staggered")
