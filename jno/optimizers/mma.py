"""Method of Moving Asymptotes — a *constrained* optimiser for jno.core.

MMA (Svanberg, *IJNME* **24**(2), 1987, 359-373) replaces the problem at each iteration with a
separable convex approximation whose asymptotes ``L`` and ``U`` adapt to the iteration history,
then solves that subproblem through its dual. It is the standard optimiser for structural design:
the objective and the constraints are expensive implicit functions of many variables, each
constraint is handled in the dual rather than as a penalty, and every iterate stays inside the box.

Why this is a sentinel rather than an optax ``GradientTransformation``: optax's ``update()`` sees
the gradient and, at most, the scalar total loss. MMA needs the *value and gradient of every
constraint separately*. Those reach it through ``Callback.on_before_update``, which is also where
the asymptote history and the multiplier state live. The same escape hatch ENGD uses.

Pair it with :func:`jno.le` / :func:`jno.ge`, which mark the entries of the ``jno.core`` list that
are constraints rather than losses -- otherwise they would be summed into the objective *and*
handled in the dual, penalising them twice::

    rho.optimizer(jno.optimizers.mma(move=0.2, lower=1e-3, upper=1.0))
    jno.core([compliance, jno.le(volume / v_star, 1.0)], domain=d).solve(200)
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ..utils.adaptive.callbacks import Callback as _Callback


class MMAOptimizer:
    """Sentinel stored on a model by ``p.optimizer(jno.optimizers.mma(...))``.

    Not an optax ``GradientTransformation``. ``jno.core.solve()`` detects it, injects
    ``optax.sgd(1.0)`` as the transform that applies the step, and prepends a single
    :class:`MMACallback` covering **every** model marked this way -- MMA solves one joint
    subproblem, because the design variables are coupled through the shared constraints.

    See :func:`mma` for the public constructor.
    """

    def __init__(
        self,
        move: float = 0.2,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        asy_init: float = 0.5,
        asy_shrink: float = 0.7,
        asy_grow: float = 1.2,
        raa: float = 1e-5,
        dual_iters: int = 400,
        move_gamma: float = 1.0,
        move_min: float = 0.0,
    ):
        self.move = float(move)
        self.move_gamma = float(move_gamma)
        self.move_min = float(move_min)
        self.lower = lower
        self.upper = upper
        self.asy_init = float(asy_init)
        self.asy_shrink = float(asy_shrink)
        self.asy_grow = float(asy_grow)
        self.raa = float(raa)
        self.dual_iters = int(dual_iters)

    def __repr__(self):
        return f"MMAOptimizer(move={self.move}, bounds=({self.lower}, {self.upper}))"


def mma(
    *,
    move: float = 0.2,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    asy_init: float = 0.5,
    asy_shrink: float = 0.7,
    asy_grow: float = 1.2,
    raa: float = 1e-5,
    dual_iters: int = 400,
    move_gamma: float = 1.0,
    move_min: float = 0.0,
) -> MMAOptimizer:
    """Return an MMA sentinel to pass to ``.optimizer()`` on a design variable.

    Args:
        move: Move limit as a fraction of the box width -- the most a variable may change in one
            iteration. The single most important knob: too large oscillates, too small crawls.
            ``0.1``-``0.2`` is usual for densities, much smaller for geometry.
        lower, upper: Box bounds — a scalar, or an array of one bound per variable. A density is
            ``lower=1e-3`` (not 0, which makes the stiffness singular) and ``upper=1.0``. A nodal
            coordinate is bounded per node, since the limit is on its movement about its OWN
            initial position: ``lower=x0 - 2``, ``upper=x0 + 2``.
        asy_init: Initial asymptote distance, as a fraction of the box width.
        asy_shrink, asy_grow: Asymptote adaptation. A variable that just reversed direction has its
            asymptotes pulled in by ``asy_shrink`` (damping the oscillation); one moving steadily
            has them pushed out by ``asy_grow`` (accelerating). This history-dependence is what
            distinguishes MMA from a fixed convex approximation.
        move_gamma: Per-iteration decay of the move limit, ``move_k = move * move_gamma**k``,
            floored at ``move_min``. MMA does not converge to a point: near the optimum the
            iterates keep chattering at the scale of the move limit, which is fine for the design
            but means a relative-change convergence test never fires. Shrinking the limit late in
            the run damps that chatter. ``1.0`` (the default) is no decay, i.e. the classic method.
        move_min: Floor on the decayed move limit, so the design never freezes outright.
        raa: Small positive term keeping the approximation strictly convex when a gradient vanishes.
        dual_iters: Projected-gradient iterations on the dual. The dual is concave in ``lambda``
            and low-dimensional (one per constraint), so this is cheap; a single constraint is
            solved by bisection instead and ignores this.

    Example -- compliance minimisation under a volume constraint::

        rho.optimizer(jno.optimizers.mma(move=0.2, lower=1e-3, upper=1.0))
        jno.core([compliance, jno.le(vol_frac, 0.4)], domain=d).solve(200)

    References:
        Svanberg, *The method of moving asymptotes -- a new method for structural optimization*,
        International Journal for Numerical Methods in Engineering **24**(2), 1987, 359-373.
    """
    return MMAOptimizer(
        move=move,
        lower=lower,
        upper=upper,
        asy_init=asy_init,
        asy_shrink=asy_shrink,
        asy_grow=asy_grow,
        raa=raa,
        dual_iters=dual_iters,
        move_gamma=move_gamma,
        move_min=move_min,
    )


class GCMMAOptimizer(MMAOptimizer):
    """Sentinel for :func:`gcmma`: MMA's, plus the inner-iteration budget. See :func:`gcmma`."""

    def __init__(self, *, max_inner: int = 10, **kw):
        super().__init__(**kw)
        self.max_inner = int(max_inner)

    def __repr__(self):
        return f"GCMMAOptimizer(move={self.move}, bounds=({self.lower}, {self.upper}), max_inner={self.max_inner})"


def gcmma(
    *,
    move: float = 0.2,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
    max_inner: int = 10,
    asy_init: float = 0.5,
    asy_shrink: float = 0.7,
    asy_grow: float = 1.2,
    dual_iters: int = 400,
    move_gamma: float = 1.0,
    move_min: float = 0.0,
) -> GCMMAOptimizer:
    """Globally convergent MMA: like :func:`mma`, but every step is feasible and lowers the objective.

    Plain MMA takes the minimiser of its convex approximation as the next design. When a constraint
    curves more sharply than the approximation (a stress or distortion measure, a p-norm), that
    step can land infeasible, and the run oscillates about the constraint. GCMMA evaluates the
    objective and the constraints at the trial point and, for each function the approximation
    underestimated there, makes that approximation more convex and solves again from the same
    design. From a feasible start every accepted design is then feasible and the objective never
    rises (Svanberg 2002), which also makes the run converge to a KKT point rather than chatter.

    The price is one extra evaluation of the objective and constraint VALUES per re-solve (no
    gradients), typically zero to a few per step. Use it like :func:`mma`, with
    :func:`jno.le` / :func:`jno.ge` in the ``jno.core`` list::

        rho.optimizer(jno.optimizers.gcmma(move=0.2, lower=1e-3, upper=1.0))
        jno.core([compliance, jno.le(vol_frac, 0.4)]).solve(200)

    ``move``, ``lower``, ``upper``, ``asy_init``, ``asy_shrink``, ``asy_grow``, ``dual_iters``,
    ``move_gamma`` and ``move_min`` are as for :func:`mma`. There is no ``raa``: GCMMA sets that term
    itself, per function.

    Args:
        max_inner: Most re-solves per step. When it runs out, the last trial point is accepted as
            it is (as in Svanberg's code) and is not guaranteed feasible; the solve logs a warning
            the first time and a count at the end.

    Every design variable in one ``jno.core`` must use the same one of ``mma`` and ``gcmma``: they
    share a single subproblem.

    References:
        Svanberg, *A class of globally convergent optimization methods based on conservative convex
        separable approximations*, SIAM Journal on Optimization **12**(2), 2002, 555-573.
    """
    return GCMMAOptimizer(
        move=move,
        lower=lower,
        upper=upper,
        max_inner=max_inner,
        asy_init=asy_init,
        asy_shrink=asy_shrink,
        asy_grow=asy_grow,
        dual_iters=dual_iters,
        move_gamma=move_gamma,
        move_min=move_min,
    )


# ---------------------------------------------------------------------------
# The subproblem
# ---------------------------------------------------------------------------
def _asymptotes(x, xold1, xold2, low, upp, xmin, xmax, k, spec_init, spec_shrink, spec_grow):
    """Svanberg eq. (11)-(12): where to put ``L`` and ``U`` this iteration.

    For the first two iterations there is no history, so the asymptotes sit a fixed fraction of the
    box width away. After that each variable is treated individually: a sign change in its last two
    steps means it is oscillating and the asymptotes close in to damp it; a consistent direction
    means it is making progress and they move out to let it travel further.
    """
    span = xmax - xmin
    if k < 2 or xold1 is None or xold2 is None:
        low = x - spec_init * span
        upp = x + spec_init * span
    else:
        sign = (x - xold1) * (xold1 - xold2)
        gamma = np.where(sign < 0.0, spec_shrink, np.where(sign > 0.0, spec_grow, 1.0))
        low = x - gamma * (xold1 - low)
        upp = x + gamma * (upp - xold1)
        # Keep the asymptotes a sane distance away: too close makes the approximation stiff and the
        # step tiny, too far makes it nearly linear and the step unreliable.
        low = np.clip(low, x - 10.0 * span, x - 0.01 * span)
        upp = np.clip(upp, x + 0.01 * span, x + 10.0 * span)
    return low, upp


def _pq(df, x, low, upp, span, raa):
    """Svanberg eq. (13)-(14): split a gradient into the two one-sided asymptote coefficients."""
    dfp = np.maximum(df, 0.0)
    dfm = np.maximum(-df, 0.0)
    p = (upp - x) ** 2 * (1.001 * dfp + 0.001 * dfm + raa / span)
    q = (x - low) ** 2 * (0.001 * dfp + 1.001 * dfm + raa / span)
    return p, q


def _primal(lam, p0, q0, P, Q, low, upp, alpha, beta):
    """The subproblem's exact minimiser for a given ``lambda`` -- Svanberg eq. (18)."""
    pl = p0 + (lam @ P if P.size else 0.0)
    ql = q0 + (lam @ Q if Q.size else 0.0)
    sp, sq = np.sqrt(np.maximum(pl, 1e-300)), np.sqrt(np.maximum(ql, 1e-300))
    x = (sp * low + sq * upp) / (sp + sq)
    return np.clip(x, alpha, beta)


def mma_subproblem(x, f0, df0, g, dg, low, upp, xmin, xmax, xold1, xold2, k, spec, dual_iters=400, with_multipliers=False):
    """One MMA step: build the convex approximation, solve its dual, return the new point.

    Args:
        x: current design ``(n,)``; ``f0``/``df0``: objective value and gradient;
        g: constraint values ``(m,)``, feasible when ``<= 0``; ``dg``: their Jacobian ``(m, n)``;
        low/upp: previous asymptotes; xold1/xold2: the two previous iterates (``None`` early on).

    Returns ``(x_new, low, upp)``, and the subproblem's multipliers ``(m,)`` after them with
    ``with_multipliers``.
    """
    low, upp = _asymptotes(x, xold1, xold2, low, upp, xmin, xmax, k, spec.asy_init, spec.asy_shrink, spec.asy_grow)
    x_new, lam = _solve_approximation(x, df0, g, dg, low, upp, xmin, xmax, spec.move, spec.raa, spec.raa, dual_iters)
    return (x_new, low, upp, lam) if with_multipliers else (x_new, low, upp)


def gcmma_step(x, f0, df0, g, dg, low, upp, xmin, xmax, xold1, xold2, k, spec, evaluate):
    """One GCMMA outer iteration: an MMA step, re-solved until its approximation is conservative.

    Svanberg, *SIAM J. Optim.* **12**(2), 2002, 555-573, in the form of Svanberg's 2007 note
    "MMA and GCMMA -- two methods for nonlinear optimization", Sec. 5. The asymptotes are set once,
    as in MMA. Each function ``f_i`` (the objective and every constraint) gets its own convexity
    term ``rho_i``, which starts at ``0.1/n * sum_j |df_i/dx_j| (xmax_j - xmin_j)`` (at least 1e-6).
    After a trial point ``x_hat`` is found, the TRUE values ``evaluate(x_hat)`` are compared with
    the approximation's. Every ``f_i`` that the approximation underestimated has its ``rho_i``
    raised by ``min(1.1 (rho_i + delta_i), 10 rho_i)``, with ``delta_i`` the shortfall divided by
    ``d(x_hat)``, the amount by which ``rho_i`` moves the approximation there. The subproblem is
    then solved again from the same ``x``.

    Once every approximation is conservative, a feasible ``x`` gives a feasible ``x_hat`` with a
    lower objective. Each inner iteration costs one evaluation of values only, no gradients.

    Args:
        f0: the objective's value at ``x`` (MMA ignores it, GCMMA needs it).
        evaluate: ``x_hat -> (f0, g)``, the true objective and constraint values at a trial point.
        spec: as for :func:`mma_subproblem`, plus ``max_inner``, the most re-solves per step.

    Returns ``(x_new, low, upp, info)``. ``info["inner"]`` is the number of re-solves, ``info["lam"]``
    the multipliers of the accepted subproblem, and
    ``info["conservative"]`` is False when ``max_inner`` ran out first: the last trial point is
    then accepted as it is, as in Svanberg's code, and is not guaranteed feasible or a descent.
    """
    x = np.asarray(x, dtype=np.float64)
    df0 = np.asarray(df0, dtype=np.float64).reshape(-1)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    m = g.size
    dg = np.asarray(dg, dtype=np.float64).reshape(m, x.size)
    span = np.maximum(xmax - xmin, 1e-12)
    low, upp = _asymptotes(x, xold1, xold2, low, upp, xmin, xmax, k, spec.asy_init, spec.asy_shrink, spec.asy_grow)

    n = x.size
    rho0 = max(1e-6, 0.1 / n * float(np.abs(df0) @ span))
    rho = np.maximum(1e-6, 0.1 / n * (np.abs(dg) @ span)) if m else np.zeros(0)
    f_at_x = np.concatenate([[float(f0)], g])
    # Svanberg's acceptance tolerance, 1e-7 on a problem scaled to O(1), here relative to each
    # function's own size so that a small objective is not held to an absolute 1e-7.
    tol = 1e-7 * np.maximum(1.0, np.abs(f_at_x))

    def increments(xh):
        """``f~_i(xh) - f_i(x)`` for every function: the approximation's change from ``x``."""
        out = np.empty(m + 1)
        for i, (dfi, ri) in enumerate(zip([df0, *dg], [rho0, *rho])):
            p, q = _pq(dfi, x, low, upp, span, ri)
            out[i] = np.sum(p / (upp - xh) + q / (xh - low)) - np.sum(p / (upp - x) + q / (x - low))
        return out

    inner = 0
    while True:
        xh, lam = _solve_approximation(x, df0, g, dg, low, upp, xmin, xmax, spec.move, rho0, rho, spec.dual_iters)
        f_new, g_new = evaluate(xh)
        true = np.concatenate([[float(f_new)], np.asarray(g_new, dtype=np.float64).reshape(-1)])
        shortfall = true - (f_at_x + increments(xh))
        bad = shortfall > tol
        if not bad.any() or inner >= spec.max_inner:
            break
        # d(xh): the change in f~_i(xh) per unit of rho_i. Nonzero whenever xh != x, and
        # xh == x cannot be optimistic (the approximation is exact there).
        d = float(np.sum((upp - low) * (xh - x) ** 2 / ((upp - xh) * (xh - low) * span)))
        delta = shortfall / max(d, 1e-300)
        both = np.concatenate([[rho0], rho])
        both = np.where(bad, np.minimum(1.1 * (both + delta), 10.0 * both), both)
        rho0, rho = float(both[0]), both[1:]
        inner += 1
    return xh, low, upp, {"inner": inner, "conservative": not bad.any(), "f0": float(true[0]), "g": true[1:], "lam": lam}


def _solve_approximation(x, df0, g, dg, low, upp, xmin, xmax, move, raa0, raa, dual_iters):
    """Minimise the convex separable approximation built at ``x`` with asymptotes ``low``/``upp``.

    ``raa0`` is the objective's convexity term and ``raa`` the constraints' (a scalar, or one per
    constraint): MMA uses one small constant for both, GCMMA raises them per function until the
    approximation is conservative (:func:`gcmma_step`).

    Returns ``(x_new, lam)``: the minimiser and the dual solution, one multiplier per constraint.
    """
    span = np.maximum(xmax - xmin, 1e-12)
    m = int(np.size(g))
    raa = np.broadcast_to(np.asarray(raa, dtype=np.float64), (m,))

    # Move limits: never leave the box, never reach the asymptote (the approximation is singular
    # there), never travel more than `move` of the box width in one iteration.
    alpha = np.maximum.reduce([xmin, low + 0.1 * (x - low), x - move * span])
    beta = np.minimum.reduce([xmax, upp - 0.1 * (upp - x), x + move * span])
    alpha = np.minimum(alpha, beta)  # a degenerate box would otherwise invert

    p0, q0 = _pq(df0, x, low, upp, span, raa0)
    if m:
        P = np.empty((m, x.size))
        Q = np.empty((m, x.size))
        for j in range(m):
            P[j], Q[j] = _pq(np.asarray(dg[j]).reshape(-1), x, low, upp, span, raa[j])
        # r_j: the constant that makes the approximation match g_j at the current point.
        b = np.array([np.sum(P[j] / (upp - x) + Q[j] / (x - low)) for j in range(m)]) - np.asarray(g)
    else:
        P = np.zeros((0, x.size))
        Q = np.zeros((0, x.size))
        b = np.zeros(0)

    def dual_grad(lam):
        xv = _primal(lam, p0, q0, P, Q, low, upp, alpha, beta)
        # dW/dlam_j = the approximated constraint at x(lam); positive means still violated.
        return np.array([np.sum(P[j] / (upp - xv) + Q[j] / (xv - low)) for j in range(m)]) - b, xv

    if m == 0:
        return _primal(np.zeros(0), p0, q0, P, Q, low, upp, alpha, beta), np.zeros(0)

    def bisect_coord(lam, j, iters=100):
        """Maximise the dual along ``lambda_j`` alone, exactly.

        The dual is concave and its ``j``-th partial is monotone decreasing in ``lambda_j`` (raising
        a multiplier can only tighten that constraint), so a sign change brackets the root and
        bisection converges without a step size to tune. ``lambda_j = 0`` when the partial is already
        negative there — an inactive constraint, its multiplier correctly at the boundary.
        """
        lam = lam.copy()
        lam[j] = 0.0
        if dual_grad(lam)[0][j] <= 0.0:
            return lam
        lo, hi = 0.0, 1.0
        lam[j] = hi
        for _ in range(200):  # grow the bracket until the constraint is satisfied
            if dual_grad(lam)[0][j] <= 0.0:
                break
            lo, hi = hi, hi * 2.0
            lam[j] = hi
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            lam[j] = mid
            if dual_grad(lam)[0][j] > 0.0:
                lo = mid
            else:
                hi = mid
        lam[j] = 0.5 * (lo + hi)
        return lam

    if m == 1:
        lam = bisect_coord(np.zeros(1), 0)
        return dual_grad(lam)[1], lam

    # Several constraints: cyclic coordinate ascent, each coordinate solved by the same exact
    # bisection. On a concave dual this converges monotonically, and unlike projected gradient it
    # has no step size to get wrong -- which matters because the constraints here are normalised to
    # wildly different scales (a volume fraction beside a p-norm of element angles).
    lam = np.zeros(m)
    sweeps = max(1, int(dual_iters) // (20 * m))
    for _ in range(sweeps):
        prev = lam.copy()
        for j in range(m):
            lam = bisect_coord(lam, j)
        if np.max(np.abs(lam - prev)) <= 1e-12 * max(1.0, float(np.max(np.abs(lam)))):
            break
    return dual_grad(lam)[1], lam


# ---------------------------------------------------------------------------
# The callback that drives it
# ---------------------------------------------------------------------------
class MMACallback(_Callback):
    """Runs the MMA subproblem between gradient computation and the parameter update.

    ``on_before_update`` is the only hook that can *actuate* a design change, so this is where MMA
    lives. It returns ``x - x_new`` as the "gradient": the injected ``optax.sgd(1.0)`` applies
    ``x + (-1) * (x - x_new) = x_new``, so the subproblem's solution lands verbatim. ENGD abuses the
    return value the same way, and ``md_decouple`` is the precedent for an update that is a new
    point rather than a scaled gradient.

    State that MMA needs and optax has no place for -- the asymptotes and the two previous iterates
    -- lives on this object, host-side, exactly as ``ENGDCallback`` keeps its Gram cache.
    """

    def __init__(self, blocks: List[Any], inequality_idx: List[int]):
        # blocks: list of (lid, MMAOptimizer) in a fixed order — the concatenated design vector.
        self._blocks = list(blocks)
        self._ineq = list(inequality_idx)
        self._k = 0
        self._low = None
        self._upp = None
        self._xold1 = None
        self._xold2 = None
        self._jac_fn = None
        self._unravel = None
        self._n_constraints = 0
        self._names: list = []
        # GCMMA: every block or none, since they share one subproblem.
        kinds = {isinstance(spec, GCMMAOptimizer) for _lid, spec in self._blocks}
        if len(kinds) > 1:
            raise ValueError(
                "jno.optimizers: the design variables mix mma() and gcmma(), but MMA solves ONE "
                "subproblem over all of them, which is either conservative (gcmma) or not (mma). "
                "Give every design variable in this jno.core the same one."
            )
        self._gcmma = kinds == {True}
        self._values_fn = None
        self._not_conservative = 0
        # The dual solution of the last subproblem, one multiplier per `jno.le` row (None before the
        # first step). `jno.core` reports it beside the constraint values.
        self.multipliers = None

    def jacobian_request(self):
        """``(rows, lids)``: the constraint rows MMA needs and the blocks they are taken w.r.t., in design-vector
        order. jno.core then computes them in the SAME program as the objective gradient (one forward pass,
        one factorisation of every FEM system per step) and hands them over as ``constraint_jacobian``;
        ``_jac_fn`` stays the fallback for the paths that do not (gradient accumulation)."""
        if not self._ineq:
            return None
        return list(self._ineq), [lid for lid, _ in self._blocks]

    # -- hooks -------------------------------------------------------------
    def on_solve_begin(self, **kw):
        """Capture what is needed to evaluate the constraint Jacobian on demand."""
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import paramax as _paramax

        from jno.utils.ad_mode import rowwise_jacobian

        compiled_fn = kw["compiled_constraints_fn"]
        # Names, so the feasibility report below can say `g_ang` rather than "row 2". A row the user
        # never named stays None and is reported by index.
        self._names = list(kw.get("constraint_names") or [])
        frozen, static = kw["frozen"], kw["static"]
        batchsize, min_consecutive = kw["batchsize"], kw["min_consecutive"]
        self._n_constraints = int(kw["n_constraints"])
        lids = [lid for lid, _ in self._blocks]

        def _losses(sub, rest, context, rng):
            trainable = {**rest, **sub}
            full = _paramax.unwrap(eqx.combine(trainable, frozen, static))
            residuals = compiled_fn(full, context, batchsize=batchsize, key=rng, min_consecutive=min_consecutive)
            return jnp.stack([jnp.mean(r) for r in residuals])

        # `rowwise_jacobian`, not `jacrev`: `jacrev` vmaps its pullback across the output rows, which
        # through a `fem.solve()` batches the adjoint `spsolve`s into block-diagonal systems (one
        # matrix copy per row, all held at once -- see `rowwise_jacobian`). Only the `jno.le` rows are
        # asked for; the objective's gradient already arrives through `grads`.
        # Prefer the unit holding ONLY the inequality rows when core compiled one: the pullbacks then
        # never walk the objective's tape, which for a PDE-constrained problem carries a sparse
        # factorisation no `jno.le` row here depends on. Correct either way -- a Jacobian over a
        # function computing exactly these rows IS these rows -- so this is purely a cost choice, and
        # it falls back cleanly for any caller that does not supply the second unit.
        ineq_fn = kw.get("compiled_inequality_fn")

        def _ineq_losses(sub, rest, context, rng):
            trainable = {**rest, **sub}
            full = _paramax.unwrap(eqx.combine(trainable, frozen, static))
            residuals = compiled_fn_ineq(full, context, batchsize=batchsize, key=rng, min_consecutive=min_consecutive)
            return jnp.stack([jnp.mean(r) for r in residuals])

        if ineq_fn is not None and self._ineq:
            compiled_fn_ineq = ineq_fn
            losses_fn, rows = _ineq_losses, list(range(len(self._ineq)))
        else:
            losses_fn, rows = _losses, list(self._ineq)

        def jac(trainable, context, rng):
            sub = {lid: trainable[lid] for lid in lids}
            rest = {k: v for k, v in trainable.items() if k not in sub}
            if not rows:
                return jnp.zeros((0, 1))
            # Differentiate w.r.t. a LIST of blocks, not the dict: a dict's leaves come out in
            # sorted-key order, while the design vector in `on_before_update` is concatenated in
            # `self._blocks` order. A list preserves that order, so the Jacobian's columns line up
            # with `x`, `df0`, `xmin` and `xmax`.
            parts = [sub[lid] for lid in lids]
            return rowwise_jacobian(lambda ps: losses_fn(dict(zip(lids, ps)), rest, context, rng), parts, rows)

        self._jac_fn = jax.jit(jac)
        if self._gcmma:
            # Values only, of every row: GCMMA's inner loop compares them with its approximation.
            self._values_fn = jax.jit(lambda trainable, context, rng: _losses(trainable, {}, context, rng))

    def _report_initial_feasibility(self, g, dg):
        """Say where every ``jno.le`` row starts, and refuse one no step can ever satisfy.

        MMA is told the constraints and left to find a feasible point, so a row that is violated at
        the initial design and CANNOT be moved looks, from the outside, exactly like a hard problem:
        the run proceeds, the objective wanders, and the diagnosis is a wall-clock away. Measured on
        a 3-D bracket whose element-quality constraints were left in the term list with the mesh
        pinned -- `g_vmx` starts at 1.734 and no design variable touches it -- the solve spent all
        250 iterations in feasibility restoration and returned a compliance of 1.65e3 against the
        4.07 the same problem gives when they are dropped.

        The values and the Jacobian are already in hand on this step, so both the report and the
        refusal are free.

        The refusal is narrow ON PURPOSE: a row is only impossible if it is violated *and* its
        gradient vanishes w.r.t. every design variable. A row that merely starts violated is
        ordinary -- that is what MMA is for -- and a row whose gradient is zero but which starts
        satisfied is harmless. Note this cannot catch a constraint that depends on the density but
        not the mesh (`g_vmx` does, through the element volumes), which is why the report exists
        alongside it rather than being replaced by it.
        """
        from jno.utils.logger import get_logger

        _log = get_logger()
        for i, gi in enumerate(np.asarray(g).reshape(-1)):
            nm = self._names[self._ineq[i]] if i < len(self._ineq) and self._ineq[i] < len(self._names) else None
            label = repr(nm) if nm else f"inequality row {i}"
            slack = float(gi)
            row = np.asarray(dg[i]).reshape(-1) if dg.size else np.zeros(0)
            dead = row.size == 0 or not np.any(np.abs(row) > 0.0)
            if slack > 0.0 and dead:
                raise ValueError(
                    f"jno.optimizers.mma: constraint {label} is violated at the initial design "
                    f"(g = {slack:+.4g}) and its gradient is identically zero w.r.t. every design "
                    f"variable, so no step can satisfy it and the solve would spend its whole "
                    f"budget in feasibility restoration. Either drop it from the term list -- an "
                    f"element-quality bound is a constant of a fixed mesh, so pinning the "
                    f"coordinates makes it one -- or make the variables it depends on trainable."
                )
            _log.info(
                f"  {label}: g = {slack:+.4g} at the initial design "
                f"({'violated' if slack > 0 else 'satisfied'}"
                f"{', and immovable' if dead else ''})"
            )

    def on_before_update(self, *, grads, trainable, context, rng, epoch, **kw):
        import jax
        from jax.flatten_util import ravel_pytree

        if self._jac_fn is None:
            return None

        # The design vector, and the objective gradient w.r.t. it. `grads` is the gradient of the
        # loss actually descended -- which, because `jno.le` entries are held out of it, is exactly
        # the objective. That is the whole reason the constraint node had to exist.
        x_parts, df0_parts, unravels = [], [], []
        for lid, _spec in self._blocks:
            xv, un = ravel_pytree(trainable[lid])
            gv, _ = ravel_pytree(grads[lid])
            x_parts.append(np.asarray(xv, dtype=np.float64))
            df0_parts.append(np.asarray(gv, dtype=np.float64))
            unravels.append((lid, un, xv.size))
        x = np.concatenate(x_parts)
        df0 = np.concatenate(df0_parts)

        # Per-block box bounds, concatenated in the same order.
        xmin, xmax = [], []
        for (lid, spec), part in zip(self._blocks, x_parts):
            lo = _bound_vector(spec.lower, part.size, lid, "lower")
            hi = _bound_vector(spec.upper, part.size, lid, "upper")
            if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
                raise ValueError(
                    "jno.optimizers.mma: lower= and upper= are required. MMA works in the design "
                    "variable's own space: the asymptotes, the move limit and the dual are all "
                    "scaled by the box width, so an unbounded variable has no step to take "
                    f"(model layer_id={lid})."
                )
            xmin.append(lo)
            xmax.append(hi)
        xmin, xmax = np.concatenate(xmin), np.concatenate(xmax)

        # Constraint VALUES come free with the hook -- they were evaluated on this step's batch, so
        # they are consistent with the gradients by construction. Only the Jacobian needs work.
        g_all = np.asarray(kw.get("individual_losses")).reshape(-1)
        if self._ineq:
            g = g_all[np.asarray(self._ineq)]
            pre = kw.get("constraint_jacobian")
            dg = np.asarray(pre if pre is not None else self._jac_fn(trainable, context, rng), dtype=np.float64)
        else:
            g, dg = np.zeros(0), np.zeros((0, x.size))

        if self._k == 0 and self._ineq:
            self._report_initial_feasibility(g, dg)

        spec = _BlendedSpec(self._blocks, [p.size for p in x_parts], k=self._k)
        if self._gcmma:
            # The objective as core descends it: the mean of the rows that are not `jno.le`.
            obj = np.ones(g_all.size, dtype=bool)
            obj[np.asarray(self._ineq, dtype=int)] = False
            ineq = np.asarray(self._ineq, dtype=int)

            def evaluate(xh):
                trial = dict(trainable)
                off = 0
                for lid, un, size in unravels:
                    trial[lid] = un(jax.numpy.asarray(xh[off : off + size]))
                    off += size
                vals = np.asarray(self._values_fn(trial, context, rng), dtype=np.float64).reshape(-1)
                return float(vals[obj].mean()), vals[ineq]

            x_new, self._low, self._upp, info = gcmma_step(
                x,
                float(g_all[obj].mean()),
                df0,
                g,
                dg,
                self._low,
                self._upp,
                xmin,
                xmax,
                self._xold1,
                self._xold2,
                self._k,
                spec,
                evaluate,
            )
            self.multipliers = np.asarray(info["lam"], dtype=np.float64)
            if not info["conservative"]:
                self._not_conservative += 1
                if self._not_conservative == 1:
                    from jno.utils.logger import get_logger

                    get_logger().warning(
                        f"jno.optimizers.gcmma: step {self._k} used all max_inner={spec.max_inner} "
                        "re-solves without a conservative approximation, so it was accepted as it is "
                        "and may be infeasible or raise the objective. Raise max_inner or lower move. "
                        "(Further such steps are counted and reported at the end.)"
                    )
        else:
            x_new, self._low, self._upp, lam = mma_subproblem(
                x,
                None,
                df0,
                g,
                dg,
                self._low,
                self._upp,
                xmin,
                xmax,
                self._xold1,
                self._xold2,
                self._k,
                spec,
                spec.dual_iters,
                with_multipliers=True,
            )
            self.multipliers = np.asarray(lam, dtype=np.float64)
        self._xold2, self._xold1 = self._xold1, x.copy()
        self._k += 1

        # sgd(1.0) applies `x - returned`, so hand it the negated step.
        delta = x - x_new
        out = dict(grads)
        off = 0
        for lid, un, size in unravels:
            out[lid] = un(jax.numpy.asarray(delta[off : off + size], dtype=jax.numpy.float32))
            off += size
        return out

    def on_training_end(self, **kw):
        if self._not_conservative:
            from jno.utils.logger import get_logger

            get_logger().warning(
                f"jno.optimizers.gcmma: {self._not_conservative} of {self._k} steps ran out of "
                "max_inner re-solves and were accepted without the feasibility and descent guarantee."
            )


def _bound_vector(bound, size: int, lid, which: str) -> np.ndarray:
    """A box bound as a ``(size,)`` vector — a scalar broadcast, or a per-variable array as given.

    Per-variable bounds are what a **nodal movement** design variable needs: the paper's movement
    limit is +/-2 about each node's OWN initial position (Jung, Yun & Kim 2026, eq. 34 and Sec. 3),
    so the box is a different interval for every node. `.trainable()` seeds the parameter at the
    absolute coordinates, so the caller spells that limit literally as ``lower=x0 - 2``,
    ``upper=x0 + 2``.
    """
    if bound is None:
        return np.full(int(size), -np.inf if which == "lower" else np.inf)
    arr = np.asarray(bound, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(int(size), float(arr[0]))
    if arr.size != int(size):
        raise ValueError(
            f"jno.optimizers.mma: {which}= has {arr.size} entries but the design variable has "
            f"{int(size)} (model layer_id={lid}). Pass a scalar, or one bound per variable."
        )
    return arr


class _BlendedSpec:
    """The subproblem's parameters for a design vector assembled from several blocks.

    The move limit is genuinely per-variable -- a density and a nodal coordinate travel at very
    different rates -- so it is carried as a vector matching ``x``. The asymptote parameters are
    properties of the algorithm rather than of any one variable, so they are taken from the first
    block; letting them differ per block would make one subproblem out of two inconsistent
    approximations.
    """

    def __init__(self, blocks, sizes, k: int = 0):
        first = blocks[0][1]
        self.asy_init = first.asy_init
        self.asy_shrink = first.asy_shrink
        self.asy_grow = first.asy_grow
        self.raa = first.raa
        self.dual_iters = first.dual_iters
        self.max_inner = getattr(first, "max_inner", 0)
        # Decayed per block, then concatenated: `move_k = max(move * gamma**k, move_min)`.
        self.move = np.concatenate(
            [
                np.full(
                    int(s),
                    max(
                        float(spec.move) * float(getattr(spec, "move_gamma", 1.0)) ** int(k),
                        float(getattr(spec, "move_min", 0.0)),
                    ),
                )
                for (_lid, spec), s in zip(blocks, sizes)
            ]
        )
