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


def mma_subproblem(x, f0, df0, g, dg, low, upp, xmin, xmax, xold1, xold2, k, spec, dual_iters=400):
    """One MMA step: build the convex approximation, solve its dual, return the new point.

    Args:
        x: current design ``(n,)``; ``f0``/``df0``: objective value and gradient;
        g: constraint values ``(m,)``, feasible when ``<= 0``; ``dg``: their Jacobian ``(m, n)``;
        low/upp: previous asymptotes; xold1/xold2: the two previous iterates (``None`` early on).

    Returns ``(x_new, low, upp)``.
    """
    span = np.maximum(xmax - xmin, 1e-12)
    low, upp = _asymptotes(x, xold1, xold2, low, upp, xmin, xmax, k, spec.asy_init, spec.asy_shrink, spec.asy_grow)

    # Move limits: never leave the box, never reach the asymptote (the approximation is singular
    # there), never travel more than `move` of the box width in one iteration.
    alpha = np.maximum.reduce([xmin, low + 0.1 * (x - low), x - spec.move * span])
    beta = np.minimum.reduce([xmax, upp - 0.1 * (upp - x), x + spec.move * span])
    alpha = np.minimum(alpha, beta)  # a degenerate box would otherwise invert

    p0, q0 = _pq(df0, x, low, upp, span, spec.raa)
    m = int(np.size(g))
    if m:
        P = np.empty((m, x.size))
        Q = np.empty((m, x.size))
        for j in range(m):
            P[j], Q[j] = _pq(np.asarray(dg[j]).reshape(-1), x, low, upp, span, spec.raa)
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
        return _primal(np.zeros(0), p0, q0, P, Q, low, upp, alpha, beta), low, upp

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
        return dual_grad(bisect_coord(np.zeros(1), 0))[1], low, upp

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
    return dual_grad(lam)[1], low, upp


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

    # -- hooks -------------------------------------------------------------
    def on_solve_begin(self, **kw):
        """Capture what is needed to evaluate the constraint Jacobian on demand."""
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import paramax as _paramax

        from jno.utils.ad_mode import rowwise_jacobian

        compiled_fn = kw["compiled_constraints_fn"]
        frozen, static = kw["frozen"], kw["static"]
        batchsize, min_consecutive = kw["batchsize"], kw["min_consecutive"]
        self._n_constraints = int(kw["n_constraints"])
        lids = [lid for lid, _ in self._blocks]

        def _losses(sub, rest, context, rng):
            trainable = {**rest, **sub}
            full = _paramax.unwrap(eqx.combine(trainable, frozen, static))
            residuals = compiled_fn(full, context, batchsize=batchsize, key=rng, min_consecutive=min_consecutive)
            return jnp.stack([jnp.mean(r) for r in residuals])

        # `rowwise_jacobian`, NOT `jacrev`: `jacrev` vmaps its pullback across the output rows, and
        # a differentiable FEM solve bottoms out in `spsolve`, which has no batching rule -- so the
        # Jacobian of a constraint that depends on `fem.solve()` cannot be taken that way at all.
        # Only the `jno.le` rows are asked for; the objective's gradient already arrives through
        # `grads`.
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
            dg = np.asarray(self._jac_fn(trainable, context, rng), dtype=np.float64)
        else:
            g, dg = np.zeros(0), np.zeros((0, x.size))

        spec = _BlendedSpec(self._blocks, [p.size for p in x_parts], k=self._k)
        x_new, self._low, self._upp = mma_subproblem(
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
        )
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
