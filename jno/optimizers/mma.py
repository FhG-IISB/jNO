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

import functools
from typing import Any, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ..utils.adaptive.callbacks import Callback as _Callback


# docstring-from: jno.optimizers.mma.mma
class MMAOptimizer:
    """Return an MMA sentinel to pass to ``.optimizer()`` on a design variable.

    Args:
        move: Move limit as a fraction of the box width -- the most a variable may change in one
            iteration. The single most important knob: too large oscillates, too small crawls.
            ``0.1``-``0.2`` is usual for densities, much smaller for geometry.
        lower: Lower box bound — a scalar, or an array of one bound per variable. A density is
            ``lower=1e-3`` (not 0, which makes the stiffness singular) and ``upper=1.0``. A nodal
            coordinate is bounded per node, since the limit is on its movement about its OWN
            initial position: ``lower=x0 - 2``, ``upper=x0 + 2``.
        upper: Upper box bound, as ``lower``.
        asy_init: Initial asymptote distance, as a fraction of the box width.
        asy_shrink: Asymptote adaptation. A variable that just reversed direction has its
            asymptotes pulled in by ``asy_shrink`` (damping the oscillation); one moving steadily
            has them pushed out by ``asy_grow`` (accelerating). This history-dependence is what
            distinguishes MMA from a fixed convex approximation.
        asy_grow: Asymptote growth factor for a variable moving steadily, see ``asy_shrink``.
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
        lower: Lower box bound — a scalar, or an array of one bound per variable. A density is
            ``lower=1e-3`` (not 0, which makes the stiffness singular) and ``upper=1.0``. A nodal
            coordinate is bounded per node, since the limit is on its movement about its OWN
            initial position: ``lower=x0 - 2``, ``upper=x0 + 2``.
        upper: Upper box bound, as ``lower``.
        asy_init: Initial asymptote distance, as a fraction of the box width.
        asy_shrink: Asymptote adaptation. A variable that just reversed direction has its
            asymptotes pulled in by ``asy_shrink`` (damping the oscillation); one moving steadily
            has them pushed out by ``asy_grow`` (accelerating). This history-dependence is what
            distinguishes MMA from a fixed convex approximation.
        asy_grow: Asymptote growth factor for a variable moving steadily, see ``asy_shrink``.
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
    """Sentinel for :func:`gcmma`: an :class:`MMAOptimizer` plus the inner-iteration budget.

    Args:
        max_inner: most re-solves of one outer step before it is accepted as is.
        **kw: the :func:`mma` options (``move``, ``lower``, ``upper``, the asymptote settings, ...).
    """

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
# The subproblem: JAX, compiled, in float64, on the device the design lives on
# ---------------------------------------------------------------------------
# Every function below is pure and traceable: one step's approximation and its dual run as ONE compiled program on the
# design's device (no copy of the design, its gradient or the constraint Jacobian to the host), and a batch of designs
# can be `jax.vmap`ped through it. The control flow of the dual (line searches, bracket growth, Newton with a line
# search) is `lax.while_loop`s, so it is jit-able and data-dependent at once. The arithmetic is float64 whatever the
# process's default: the asymptotes and the dual's line searches need it (`jax.enable_x64` around every entry point).

#: the dual's work at its last call (`_solve_approximation`): row evaluations of the line searches, evaluations of the
#: primal minimiser, Newton steps and sweeps -- counters, not timings, read by the tests
last_dual_stats: dict = {}


def _f64(a, like=None):
    """``a`` as float64 (inside ``jax.enable_x64``), on ``like``'s device when given: every input of one compiled call
    on the same device and committed alike, so a host array (a carried history, a bound) does not
    compile the step a second time beside the device arrays it otherwise gets."""
    a = jnp.asarray(a, dtype=jnp.float64)
    if like is not None and getattr(like, "sharding", None) is not None and getattr(a, "sharding", None) != like.sharding:
        a = jax.device_put(a, like.sharding)
    return a


@functools.partial(jax.jit, static_argnames=("first",))
def _asymptotes_jit(x, xold1, xold2, low, upp, xmin, xmax, init, shrink, grow, first):
    span = xmax - xmin
    if first:
        return x - init * span, x + init * span
    sign = (x - xold1) * (xold1 - xold2)
    gamma = jnp.where(sign < 0.0, shrink, jnp.where(sign > 0.0, grow, 1.0))
    low = x - gamma * (xold1 - low)
    upp = x + gamma * (upp - xold1)
    # Keep the asymptotes a sane distance away: too close makes the approximation stiff and the
    # step tiny, too far makes it nearly linear and the step unreliable.
    low = jnp.clip(low, x - 10.0 * span, x - 0.01 * span)
    upp = jnp.clip(upp, x + 0.01 * span, x + 10.0 * span)
    return low, upp


def _asymptotes(x, xold1, xold2, low, upp, xmin, xmax, k, spec_init, spec_shrink, spec_grow):
    """Svanberg eq. (11)-(12): where to put ``L`` and ``U`` this iteration.

    For the first two iterations there is no history, so the asymptotes sit a fixed fraction of the
    box width away. After that each variable is treated individually: a sign change in its last two
    steps means it is oscillating and the asymptotes close in to damp it; a consistent direction
    means it is making progress and they move out to let it travel further.
    """
    first = k < 2 or xold1 is None or xold2 is None or low is None or upp is None
    with jax.enable_x64(True):
        x = _f64(x)
        z = jnp.zeros_like(x)
        args = (x, z, z, z, z) if first else (x, _f64(xold1, x), _f64(xold2, x), _f64(low, x), _f64(upp, x))
        return _asymptotes_jit(*args, _f64(xmin, x), _f64(xmax, x), spec_init, spec_shrink, spec_grow, first)


def _pq(df, x, low, upp, span, raa):
    """Svanberg eq. (13)-(14): split a gradient into the two one-sided asymptote coefficients (rows of ``df`` and of
    a ``raa`` column at once when ``df`` is ``(m, n)``)."""
    dfp = jnp.maximum(df, 0.0)
    dfm = jnp.maximum(-df, 0.0)
    p = (upp - x) ** 2 * (1.001 * dfp + 0.001 * dfm + raa / span)
    q = (x - low) ** 2 * (0.001 * dfp + 1.001 * dfm + raa / span)
    return p, q


def _primal(lam, p0, q0, P, Q, low, upp, alpha, beta):
    """The subproblem's exact minimiser for a given ``lambda`` -- Svanberg eq. (18)."""
    pl, ql = p0 + lam @ P, q0 + lam @ Q
    sp, sq = jnp.sqrt(jnp.maximum(pl, 1e-300)), jnp.sqrt(jnp.maximum(ql, 1e-300))
    return jnp.clip((sp * low + sq * upp) / (sp + sq), alpha, beta)


def _dual(p0, q0, P, Q, b, low, upp, alpha, beta, lam0, sweeps, has_start):
    """Maximise the concave dual of the separable approximation over ``lambda >= 0``; ``(lambda, unbounded, stats)``.

    The method of the host solver this replaces, step for step:
      * ``m = 1``: one exact line search along the multiplier (below);
      * a start (the last step's multipliers): projected Newton from it, usually done in a step or two;
      * otherwise, or when that does not converge: sweeps of exact line searches along each multiplier, each
        followed by projected Newton, until Newton converges or the multipliers stop changing.

    A line search finds the root of the row's partial, monotone decreasing in its multiplier: ``0`` when the partial
    is already negative there (an inactive row), else a bracket grown by doubling and the root by the Illinois method
    (regula falsi that halves a stuck end; it keeps the bracket, so it converges always, superlinearly), to the last
    ulp. A row whose partial stays positive after 200 doublings cannot be met within the move limit: it is marked
    unbounded, keeps the top of its bracket, and Newton leaves it alone. Projected Newton (Bertsekas, SIAM J. Control
    Optim. 20, 1982, 221): the Hessian ``-sum_i G_ji G_ki / psi_i`` over the variables away from their move limits,
    scaled by its diagonal, on the rows that are positive or pushing out; an Armijo backtracking line search."""
    m = b.shape[0]
    eps = jnp.finfo(jnp.float64).eps

    def row_minus_b(t, j, pl, ql):
        Pj, Qj = P[j], Q[j]
        sp, sq = jnp.sqrt(jnp.maximum(pl + t * Pj, 1e-300)), jnp.sqrt(jnp.maximum(ql + t * Qj, 1e-300))
        x = jnp.clip((sp * low + sq * upp) / (sp + sq), alpha, beta)
        return Pj @ (1.0 / (upp - x)) + Qj @ (1.0 / (x - low)) - b[j]

    def coord(lam, unb, j):
        """the exact line search along ``lambda_j``: ``(lambda, unbounded, row evaluations)``"""
        hi0 = jnp.maximum(1.0, 2.0 * lam[j])
        lam = lam.at[j].set(0.0)
        unb = unb.at[j].set(False)
        pl, ql = p0 + lam @ P, q0 + lam @ Q
        f = lambda t: row_minus_b(t, j, pl, ql)  # noqa: E731
        f0 = f(0.0)

        def search(_):
            def grow(s):
                _lo, hi, _fhi, it = s
                return hi, 2.0 * hi, f(2.0 * hi), it + 1

            lo, hi, fhi, nb = lax.while_loop(lambda s: (s[2] > 0.0) & (s[3] < 200), grow, (0.0 * hi0, hi0, f(hi0), 0))
            out_of_reach = fhi > 0.0
            flo = jnp.where(lo == 0.0, f0, f(lo))

            def illinois(s):
                a, fa, c, fc, side, it, _done = s
                t = (a * fc - c * fa) / (fc - fa)
                t = jnp.where(jnp.isfinite(t) & (t > a) & (t < c), t, 0.5 * (a + c))
                ft = f(t)
                pos = ft > 0.0
                a2, fa2 = jnp.where(pos, t, a), jnp.where(pos, ft, jnp.where(side < 0, 0.5 * fa, fa))
                c2, fc2 = jnp.where(pos, c, t), jnp.where(pos, jnp.where(side > 0, 0.5 * fc, fc), ft)
                side2 = jnp.where(pos, 1, -1)
                done = (ft == 0.0) | (c2 - a2 <= 4.0 * eps * jnp.maximum(jnp.abs(a2), jnp.abs(c2)))
                c2 = jnp.where(ft == 0.0, t, c2)
                return a2, fa2, c2, fc2, side2, it + 1, done

            init = (lo, flo, hi, fhi, 0, 0, out_of_reach)
            _a, _fa, root, _fr, _side, nr, _d = lax.while_loop(lambda s: (~s[6]) & (s[5] < 300), illinois, init)
            # the f <= 0 end of the bracket: the row is met there (the root to the last ulp)
            return jnp.where(out_of_reach, hi, root), out_of_reach, 2 + nb + nr

        t, unbounded, n = lax.cond(f0 > 0.0, search, lambda _: (0.0 * hi0, False, 1), None)
        return lam.at[j].set(t), unb.at[j].set(unbounded), n

    def at(lam_):
        x = _primal(lam_, p0, q0, P, Q, low, upp, alpha, beta)
        ru, rl = 1.0 / (upp - x), 1.0 / (x - low)
        rows = P @ ru + Q @ rl
        return x, ru, rl, rows, p0 @ ru + q0 @ rl + lam_ @ (rows - b)

    def newton(lam, frozen, max_iter=20, tol=1e-11):
        """projected Newton from ``lam``: ``(lambda, converged, steps, primal evaluations)``"""

        def body(s):
            lam, x, ru, rl, rows, W, _done, _failed, it, n_at = s
            grad = rows - b
            keep = ~frozen & ((lam > 0.0) | (grad > 0.0))
            conv = ~jnp.any(keep & (jnp.abs(grad) > tol * (rows + jnp.abs(b))))
            G = P * ru**2 - Q * rl**2
            away = ((x > alpha) & (x < beta)) / (2.0 * ((p0 + lam @ P) * ru**3 + (q0 + lam @ Q) * rl**3))
            H = (G * away) @ G.T
            d = jnp.sqrt(jnp.maximum(jnp.diag(H), 1e-300))  # scaled: the rows' sizes differ by many orders
            kk = keep[:, None] & keep[None, :]
            A = jnp.where(kk, H / jnp.outer(d, d) + 1e-13 * jnp.eye(m), jnp.eye(m))
            step = jnp.where(keep, jnp.linalg.solve(A, jnp.where(keep, grad / d, 0.0)) / d, 0.0)
            finite = jnp.all(jnp.isfinite(step))

            def armijo(ls):
                tt, _trial, _vals, it_ls, _ok = ls
                trial = jnp.where(keep, jnp.maximum(lam + tt * step, 0.0), lam)
                vals = at(trial)
                ok = vals[4] >= W + 1e-4 * (grad @ (trial - lam))
                return 0.5 * tt, trial, vals, it_ls + 1, ok

            vals0 = (x, ru, rl, rows, W)
            _t, trial, vals, n_ls, ok = lax.while_loop(
                lambda ls: (~ls[4]) & (ls[3] < 25) & ~conv & finite, armijo, (1.0, lam, vals0, 0, False)
            )
            failed = ~conv & (~finite | ~ok)
            moved = ~conv & ~failed
            lam2 = jnp.where(moved, trial, lam)
            x2, ru2, rl2, rows2, W2 = (jnp.where(moved, a_, b_) for a_, b_ in zip(vals, vals0))
            return lam2, x2, ru2, rl2, rows2, W2, conv, failed, it + 1, n_at + n_ls

        x, ru, rl, rows, W = at(lam)
        init = (lam, x, ru, rl, rows, W, False, False, 0, 1)
        out = lax.while_loop(lambda s: (~s[6]) & (~s[7]) & (s[8] < max_iter + 1), body, init)
        return out[0], out[6], out[8], out[9]

    zeros_b = jnp.zeros(m, bool)
    if m == 1:
        lam, unb, n_row = coord(
            jnp.zeros(1) if lam0 is None else jnp.maximum(jnp.nan_to_num(lam0, nan=0.0), 0.0), zeros_b, 0
        )
        return lam, unb, {"rows": n_row, "primal": 0, "newton": 0, "sweeps": 1}

    def sweeps_from_zero(_):
        def sweep(s):
            lam, unb, _done, _stop, k, n_row, n_newton, n_at = s
            prev = lam

            def one(j, c):
                lam_, unb_, nr_ = c
                lam_, unb_, n = coord(lam_, unb_, j)
                return lam_, unb_, nr_ + n

            lam, unb, n_row = lax.fori_loop(0, m, one, (lam, unb, n_row))
            lam, done, steps, n_at_ = newton(lam, unb)
            still = jnp.max(jnp.abs(lam - prev)) <= 1e-12 * jnp.maximum(1.0, jnp.max(jnp.abs(lam)))
            return lam, unb, done, done | still, k + 1, n_row, n_newton + steps, n_at + n_at_

        init = (jnp.zeros(m), zeros_b, False, False, 0, 0, 0, 0)
        lam, unb, _done, _stop, k, n_row, n_newton, n_at = lax.while_loop(lambda s: (~s[3]) & (s[4] < sweeps), sweep, init)
        return lam, unb, n_row, n_newton, n_at, k

    if not has_start:
        lam, unb, n_row, n_newton, n_at, k = sweeps_from_zero(None)
        return lam, unb, {"rows": n_row, "primal": n_at, "newton": n_newton, "sweeps": k}

    start_ok = jnp.all(jnp.isfinite(lam0))
    lam_s, done_s, steps_s, n_at_s = newton(jnp.where(start_ok, jnp.maximum(lam0, 0.0), 0.0), zeros_b)
    use_start = start_ok & done_s
    lam, unb, n_row, n_newton, n_at, k = lax.cond(use_start, lambda _: (lam_s, zeros_b, 0, 0, 0, 0), sweeps_from_zero, None)
    stats = {"rows": n_row, "primal": n_at + n_at_s, "newton": n_newton + steps_s, "sweeps": k}
    return lam, unb, stats


@functools.partial(jax.jit, static_argnames=("sweeps", "has_start"))
def _approximation_jit(x, df0, g, dg, low, upp, xmin, xmax, move, raa0, raa, lam0, sweeps, has_start):
    span = jnp.maximum(xmax - xmin, 1e-12)
    m = g.shape[0]
    # Move limits: never leave the box, never reach the asymptote (the approximation is singular
    # there), never travel more than `move` of the box width in one iteration.
    alpha = jnp.maximum(jnp.maximum(xmin, low + 0.1 * (x - low)), x - move * span)
    beta = jnp.minimum(jnp.minimum(xmax, upp - 0.1 * (upp - x)), x + move * span)
    alpha = jnp.minimum(alpha, beta)  # a degenerate box would otherwise invert
    p0, q0 = _pq(df0, x, low, upp, span, raa0)
    P, Q = _pq(dg, x, low, upp, span, jnp.broadcast_to(raa, (m,))[:, None])
    # r_j: the constant that makes the approximation match g_j at the current point.
    b = P @ (1.0 / (upp - x)) + Q @ (1.0 / (x - low)) - g
    if m == 0:
        return _primal(jnp.zeros(0), p0, q0, P, Q, low, upp, alpha, beta), jnp.zeros(0), {}
    lam, unb, stats = _dual(p0, q0, P, Q, b, low, upp, alpha, beta, lam0, sweeps, has_start)
    x_new = _primal(lam, p0, q0, P, Q, low, upp, alpha, beta)
    return x_new, jnp.where(unb, jnp.nan, lam), stats


def _solve_approximation(x, df0, g, dg, low, upp, xmin, xmax, move, raa0, raa, dual_iters, lam0=None):
    """Minimise the convex separable approximation built at ``x`` with asymptotes ``low``/``upp``.

    ``raa0`` is the objective's convexity term and ``raa`` the constraints' (a scalar, or one per
    constraint): MMA uses one small constant for both, GCMMA raises them per function until the
    approximation is conservative (:func:`gcmma_step`). ``lam0``: where to start the dual (the last subproblem's
    multipliers: one step changes them little); the answer does not depend on it.

    Returns ``(x_new, lam)`` as float64 device arrays: the minimiser and the dual solution, one multiplier per
    constraint. A row the subproblem cannot meet within the move limit has no multiplier (the dual is unbounded along
    it): its entry is NaN, and ``x_new`` is still the point the bracket search ended on, as far towards meeting it as
    the move limit allows. One compiled program (:func:`_dual`); its work counters land in ``last_dual_stats``.
    """
    with jax.enable_x64(True):
        g = _f64(g).reshape(-1)
        m = int(g.shape[0])
        x = _f64(x).reshape(-1)
        dg = _f64(dg).reshape(m, x.shape[0])
        has_start = lam0 is not None and m > 0
        x_new, lam, stats = _approximation_jit(
            x,
            _f64(df0, x).reshape(-1),
            _f64(g, x),
            _f64(dg, x),
            _f64(low, x),
            _f64(upp, x),
            _f64(xmin, x),
            _f64(xmax, x),
            _f64(move, x),
            _f64(raa0, x),
            _f64(raa, x),
            _f64(lam0, x).reshape(-1) if has_start else None,
            sweeps=max(1, int(dual_iters) // (20 * max(m, 1))),
            has_start=has_start,
        )
    last_dual_stats.clear()
    last_dual_stats.update({k_: int(v_) for k_, v_ in stats.items()})
    return x_new, lam


def mma_subproblem(
    x, f0, df0, g, dg, low, upp, xmin, xmax, xold1, xold2, k, spec, dual_iters=400, with_multipliers=False, lam0=None
):
    """One MMA step: build the convex approximation, solve its dual, return the new point.

    Args:
        x: current design ``(n,)``; ``f0``/``df0``: objective value and gradient;
        g: constraint values ``(m,)``, feasible when ``<= 0``; ``dg``: their Jacobian ``(m, n)``;
        low/upp: previous asymptotes; xold1/xold2: the two previous iterates (``None`` early on);
        lam0: a start for the dual, the last subproblem's multipliers (the answer does not depend on it).

    Returns ``(x_new, low, upp)`` (float64 device arrays), and the subproblem's multipliers ``(m,)`` after them with
    ``with_multipliers``.
    """
    low, upp = _asymptotes(x, xold1, xold2, low, upp, xmin, xmax, k, spec.asy_init, spec.asy_shrink, spec.asy_grow)
    x_new, lam = _solve_approximation(x, df0, g, dg, low, upp, xmin, xmax, spec.move, spec.raa, spec.raa, dual_iters, lam0)
    return (x_new, low, upp, lam) if with_multipliers else (x_new, low, upp)


@jax.jit
def _approximation_change(DF, rho, x, xh, low, upp, span):
    """``f~_i(xh) - f_i(x)`` for every row of ``DF`` (the objective's gradient and the constraints'), each with its
    convexity term ``rho_i``: the approximation's change from ``x``."""
    p, q = _pq(DF, x, low, upp, span, rho[:, None])
    return (p / (upp - xh) + q / (xh - low)).sum(1) - (p / (upp - x) + q / (x - low)).sum(1)


def gcmma_step(x, f0, df0, g, dg, low, upp, xmin, xmax, xold1, xold2, k, spec, evaluate, lam0=None):
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
    The vector work runs on the design's device; the loop (one evaluation of the true values per re-solve) is host
    control flow.
    """
    low, upp = _asymptotes(x, xold1, xold2, low, upp, xmin, xmax, k, spec.asy_init, spec.asy_shrink, spec.asy_grow)
    with jax.enable_x64(True):
        x = _f64(x).reshape(-1)
        df0 = _f64(df0).reshape(-1)
        g = _f64(g).reshape(-1)
        m = int(g.shape[0])
        n = int(x.shape[0])
        dg = _f64(dg).reshape(m, n)
        xmin, xmax = _f64(xmin), _f64(xmax)
        span = jnp.maximum(xmax - xmin, 1e-12)
        DF = jnp.concatenate([df0[None, :], dg], axis=0)
        rho = np.maximum(1e-6, 0.1 / n * np.asarray(jnp.abs(DF) @ span))  # (m + 1,): the objective's, then each row's
        f_at_x = np.concatenate([[float(f0)], np.asarray(g)])
        # Svanberg's acceptance tolerance, 1e-7 on a problem scaled to O(1), here relative to each
        # function's own size so that a small objective is not held to an absolute 1e-7.
        tol = 1e-7 * np.maximum(1.0, np.abs(f_at_x))

        inner, lam = 0, lam0
        while True:
            xh, lam = _solve_approximation(
                x, df0, g, dg, low, upp, xmin, xmax, spec.move, rho[0], rho[1:], spec.dual_iters, lam
            )
            f_new, g_new = evaluate(xh)
            true = np.concatenate([[float(f_new)], np.asarray(g_new, dtype=np.float64).reshape(-1)])
            shortfall = true - (f_at_x + np.asarray(_approximation_change(DF, _f64(rho), x, xh, low, upp, span)))
            bad = shortfall > tol
            if not bad.any() or inner >= spec.max_inner:
                break
            # d(xh): the change in f~_i(xh) per unit of rho_i. Nonzero whenever xh != x, and
            # xh == x cannot be optimistic (the approximation is exact there).
            d = float(jnp.sum((upp - low) * (xh - x) ** 2 / ((upp - xh) * (xh - low) * span)))
            delta = shortfall / max(d, 1e-300)
            rho = np.where(bad, np.minimum(1.1 * (rho + delta), 10.0 * rho), rho)
            inner += 1
    info = {"inner": inner, "conservative": not bad.any(), "f0": float(true[0]), "g": true[1:], "lam": lam}
    info["optimistic"] = np.flatnonzero(bad)  # the functions still underestimated: 0 the objective, i the i-th row
    return xh, low, upp, info


# ---------------------------------------------------------------------------
# The callback that drives it
# ---------------------------------------------------------------------------
def _mma_programs(compiled_fn, ineq_fn, ineq_rows, lids, batchsize, min_consecutive, n_values):
    """MMA's constraint-Jacobian program (and, for ``n_values > 0``, GCMMA's program of the first ``n_values`` row
    values) as jitted functions of ``(trainable, frozen, static, context, rng)``: `MMACallback.on_solve_begin` keeps
    them across solves."""
    import equinox as eqx
    import jax.numpy as jnp
    import paramax as _paramax

    from jno.utils.ad_mode import rowwise_jacobian

    def reduced(fn, rows, trainable, frozen, static, context, rng):
        full = _paramax.unwrap(eqx.combine(trainable, frozen, static))
        residuals = fn(full, context, batchsize=batchsize, key=rng, min_consecutive=min_consecutive)
        # Only the asked-for rows are reduced, so whatever the others need -- the objective's own
        # solve, typically the expensive part of the model -- is dead code and is dropped when
        # this compiles. Stacking every row and pulling back a zero cotangent does NOT drop it:
        # the zero is a computed array, so the adjoint solve runs on it and needs the forward one.
        # Measured on an inductance design: the field solve re-ran every step, 190 ms of 550 ms.
        return jnp.stack([jnp.mean(residuals[i]) for i in rows])

    # `rowwise_jacobian`, not `jacrev`: `jacrev` vmaps its pullback across the output rows, which
    # through a `fem.solve()` batches the adjoint `spsolve`s into block-diagonal systems (one
    # matrix copy per row, all held at once -- see `rowwise_jacobian`). Only the `jno.le` rows are
    # asked for; the objective's gradient already arrives through `grads`.
    # Prefer the unit holding ONLY the inequality rows when core compiled one (``ineq_fn``): the pullbacks
    # then never walk the objective's tape, which for a PDE-constrained problem carries a sparse
    # factorisation no `jno.le` row here depends on. Correct either way -- a Jacobian over a
    # function computing exactly these rows IS these rows -- so this is purely a cost choice.
    fn, rows = (compiled_fn, list(ineq_rows)) if ineq_fn is None else (ineq_fn, list(range(len(ineq_rows))))
    stacked = list(range(len(rows)))  # the reduced rows are numbered 0..n-1

    def jac(trainable, frozen, static, context, rng):
        if not rows:
            return jnp.zeros((0, 1))
        sub = {lid: trainable[lid] for lid in lids}
        rest = {k: v for k, v in trainable.items() if k not in sub}
        # Differentiate w.r.t. a LIST of blocks, not the dict: a dict's leaves come out in
        # sorted-key order, while the design vector in `on_before_update` is concatenated in
        # `self._blocks` order. A list preserves that order, so the Jacobian's columns line up
        # with `x`, `df0`, `xmin` and `xmax`.
        parts = [sub[lid] for lid in lids]
        return rowwise_jacobian(
            lambda ps: reduced(fn, rows, {**rest, **dict(zip(lids, ps))}, frozen, static, context, rng), parts, stacked
        )

    def values(trainable, frozen, static, context, rng):
        # Values only, of every row: GCMMA's inner loop compares them with its approximation.
        return reduced(compiled_fn, range(n_values), trainable, frozen, static, context, rng)

    return eqx.filter_jit(jac), (eqx.filter_jit(values) if n_values else None)


class MMACallback(_Callback):
    """Runs the MMA subproblem between gradient computation and the parameter update.

    ``on_before_update`` is the only hook that can *actuate* a design change, so this is where MMA
    lives. It returns ``x - x_new`` as the "gradient": the injected ``optax.sgd(1.0)`` applies
    ``x + (-1) * (x - x_new) = x_new``, so the subproblem's solution lands verbatim. ENGD abuses the
    return value the same way, and ``md_decouple`` is the precedent for an update that is a new
    point rather than a scaled gradient.

    State that MMA needs and optax has no place for -- the asymptotes and the two previous iterates
    -- lives on this object, as device arrays where the design lives (the subproblem is one compiled program on them;
    only the m constraint values and multipliers come to the host), as ``ENGDCallback`` keeps its Gram cache. ``jno.core`` builds one
    per ``solve`` call and carries the previous one's history into it (:meth:`continue_from`), so a second
    ``solve`` continues the first.

    Args:
        blocks: ``(layer_id, MMAOptimizer)`` pairs, in the order their variables are concatenated into
            the design vector.
        inequality_idx: rows of ``individual_losses`` that are ``jno.le`` / ``jno.ge`` constraints.
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
        self._resolves = 0  # GCMMA: re-solves over all steps
        self._held_back = {}  # GCMMA: function name -> steps it kept from being conservative
        # The dual solution of the last subproblem, one multiplier per `jno.le` row (None before the
        # first step). `jno.core` reports it beside the constraint values.
        self.multipliers = None
        self._warned_unreachable = False

    # -- history across solve() calls ------------------------------------------------------------
    _HISTORY = ("_k", "_low", "_upp", "_xold1", "_xold2", "multipliers")

    def continue_from(self, prev) -> bool:
        """Carry ``prev``'s history (the asymptotes, the two previous iterates, the step count, the multipliers) into
        this callback, so a second ``jno.core.solve()`` continues the first as an optax
        state does; returns whether it was carried.

        A solve that trains other blocks than ``prev`` (one unfrozen or frozen in between) keeps the history of the
        blocks both train; a block new to this solve starts as at a first step (its asymptotes ``asy_init`` of its
        box away), and the step count carries on, so ``move_gamma`` keeps shrinking the move limit from where it
        was. The multipliers carry only for the same constraint rows. A design vector that changed size since is caught at the next step (`on_before_update`), which
        restarts."""
        if prev is None or prev is self:
            return False
        if prev._gcmma != self._gcmma:
            return False
        same_rows = list(prev._ineq) == self._ineq
        if [lid for lid, _ in prev._blocks] == [lid for lid, _ in self._blocks]:
            if not same_rows:
                return False
            for name in self._HISTORY:
                setattr(self, name, getattr(prev, name))
        else:
            sizes = getattr(prev, "_sizes", None)
            if (
                prev._low is None
                or sizes is None
                or not {lid for lid, _ in prev._blocks} & {lid for lid, _ in self._blocks}
            ):
                return False
            segs, off = {}, 0
            for lid, size in sizes:  # each block's slice of the previous design vector's history
                segs[lid] = tuple(
                    None if v is None else v[off : off + size] for v in (prev._low, prev._upp, prev._xold1, prev._xold2)
                )
                off += size
            self._pending = {lid: segs[lid] for lid, _ in self._blocks if lid in segs}
            self._k = prev._k
            self.multipliers = prev.multipliers if same_rows else None
        self._not_conservative = prev._not_conservative
        self._resolves = getattr(prev, "_resolves", 0)
        self._held_back = dict(getattr(prev, "_held_back", {}))
        self._warned_unreachable = prev._warned_unreachable
        return True

    def _assemble_pending(self, x_parts) -> None:
        """The history carried per block (`continue_from` across a change of blocks), put together for this solve's
        design vector: a block new to it gets ``low/upp = x -+ asy_init * span`` and both previous iterates at ``x``,
        so Svanberg's update keeps exactly those asymptotes at its next step (no sign change, factor 1)."""
        pending, self._pending = self._pending, None
        lows, upps, x1s, x2s = [], [], [], []
        with jax.enable_x64(True):
            for (lid, spec), part in zip(self._blocks, x_parts):
                xv = _f64(part)
                got = pending.get(lid)
                if got is not None and np.size(got[0]) == xv.size and all(v is not None for v in got):
                    lo, up, x1, x2 = (_f64(v) for v in got)
                else:
                    span = _f64(_bound_vector(spec.upper, xv.size, lid, "upper")) - _f64(
                        _bound_vector(spec.lower, xv.size, lid, "lower")
                    )
                    lo, up, x1, x2 = xv - spec.asy_init * span, xv + spec.asy_init * span, xv, xv
                lows.append(lo), upps.append(up), x1s.append(x1), x2s.append(x2)
            self._low, self._upp = jnp.concatenate(lows), jnp.concatenate(upps)
            self._xold1, self._xold2 = jnp.concatenate(x1s), jnp.concatenate(x2s)

    def _restart(self, why: str) -> None:
        from jno.utils.logger import get_logger

        if self._k:
            get_logger().info(f"  MMA: {why}; its asymptote history restarts.")
        self._k, self._low, self._upp, self._xold1, self._xold2 = 0, None, None, None, None

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
        """Capture what is needed to evaluate the constraint Jacobian on demand.

        Keyword Args:
            The solve context of :meth:`Callback.on_solve_begin`; this hook reads
            ``compiled_constraints_fn``, ``n_constraints``, ``batchsize``, ``frozen``, ``static``,
            ``min_consecutive`` and (optionally) ``constraint_names``.
        """
        compiled_fn = kw["compiled_constraints_fn"]
        # Names, so the feasibility report below can say `g_ang` rather than "row 2". A row the user
        # never named stays None and is reported by index.
        self._names = list(kw.get("constraint_names") or [])
        frozen, static = kw["frozen"], kw["static"]
        batchsize, min_consecutive = kw["batchsize"], kw["min_consecutive"]
        self._n_constraints = int(kw["n_constraints"])
        lids = [lid for lid, _ in self._blocks]
        ineq_fn = kw.get("compiled_inequality_fn")
        ineq_fn = ineq_fn if ineq_fn is not None and self._ineq else None
        # The programs are kept across solves, in jno.core's program set (kept with its step program): a closure made
        # anew per solve would be compiled anew per solve. What changes between solves -- the frozen values and the
        # static parts -- are ARGUMENTS of the cached programs. A caller without a program set makes them per solve.
        key = ("mma", id(compiled_fn), id(ineq_fn), tuple(lids), tuple(self._ineq), batchsize, min_consecutive)
        key += (self._n_constraints if self._gcmma else 0,)
        cache = kw.get("program_cache")
        cache = {} if cache is None else cache
        programs = cache.get(key)
        if programs is None:
            programs = cache[key] = _mma_programs(
                compiled_fn, ineq_fn, list(self._ineq), lids, batchsize, min_consecutive, key[-1]
            )
        jac_p, values_p = programs
        self._jac_fn = lambda trainable, context, rng: jac_p(trainable, frozen, static, context, rng)
        if values_p is not None:
            self._values_fn = lambda trainable, context, rng: values_p(trainable, frozen, static, context, rng)

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
        alive = np.asarray(jnp.any(jnp.abs(dg) > 0.0, axis=1)) if dg.size else np.zeros(0, bool)  # m flags, not rows
        for i, gi in enumerate(np.asarray(g).reshape(-1)):
            nm = self._names[self._ineq[i]] if i < len(self._ineq) and self._ineq[i] < len(self._names) else None
            label = repr(nm) if nm else f"inequality row {i}"
            slack = float(gi)
            dead = i >= alive.size or not bool(alive[i])
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
        # Everything the size of the design stays where it is (on the GPU for a GPU run): the design, the objective's
        # gradient, the constraint Jacobian, the asymptotes and the iterates are device arrays, and the subproblem is
        # one compiled program on them. Only the m constraint values and multipliers come to the host, for the
        # decisions taken here (the reports).
        x_parts, df0_parts, unravels = [], [], []
        with jax.enable_x64(True):
            for lid, _spec in self._blocks:
                xv, un = ravel_pytree(trainable[lid])
                gv, _ = ravel_pytree(grads[lid])
                x_parts.append(_f64(xv))
                df0_parts.append(_f64(gv))
                unravels.append((lid, un, xv.size, xv.dtype))
            x = jnp.concatenate(x_parts)
            df0 = jnp.concatenate(df0_parts)
        self._sizes = [(lid, int(part.size)) for (lid, _spec), part in zip(self._blocks, x_parts)]
        if getattr(self, "_pending", None) is not None:  # history carried per block from a solve with other blocks
            self._assemble_pending(x_parts)
        if self._low is not None and np.size(self._low) != x.size:  # carried from a design of another layout
            self._restart(f"the design vector has {x.size} values, its history {np.size(self._low)}")

        # Per-block box bounds, concatenated in the same order: built once per layout and kept on the device.
        box_key = tuple(
            (lid, id(spec.lower), id(spec.upper), int(part.size)) for (lid, spec), part in zip(self._blocks, x_parts)
        )
        box = getattr(self, "_box", None)
        if box is None or box[0] != box_key:
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
            with jax.enable_x64(True):
                box = self._box = (box_key, _f64(np.concatenate(xmin)), _f64(np.concatenate(xmax)))
        xmin, xmax = box[1], box[2]

        # Constraint VALUES come free with the hook -- they were evaluated on this step's batch, so
        # they are consistent with the gradients by construction. Only the Jacobian needs work.
        g_all = np.asarray(kw.get("individual_losses")).reshape(-1)
        g = g_all[np.asarray(self._ineq)] if self._ineq else np.zeros(0)
        with jax.enable_x64(True):
            if self._ineq:
                pre = kw.get("constraint_jacobian")
                dg = _f64(pre if pre is not None else self._jac_fn(trainable, context, rng)).reshape(len(self._ineq), -1)
            else:
                dg = jnp.zeros((0, x.size))
        if self._k == 0 and self._ineq:
            self._report_initial_feasibility(g, dg)
        low_in, upp_in, xold1, xold2, k = self._low, self._upp, self._xold1, self._xold2, self._k

        spec = _BlendedSpec(self._blocks, [p.size for p in x_parts], k=k)
        if self._gcmma:
            # The objective as core descends it: the mean of the rows that are not `jno.le`.
            obj = np.ones(g_all.size, dtype=bool)
            obj[np.asarray(self._ineq, dtype=int)] = False
            ineq = np.asarray(self._ineq, dtype=int)

            def evaluate(xh):
                trial = dict(trainable)
                off = 0
                for lid, un, size, dt in unravels:
                    trial[lid] = un(jnp.asarray(xh[off : off + size], dtype=dt))
                    off += size
                vals = np.asarray(self._values_fn(trial, context, rng), dtype=np.float64).reshape(-1)
                return float(vals[obj].mean()), vals[ineq]

            x_new, self._low, self._upp, info = gcmma_step(
                x,
                float(g_all[obj].mean()),
                df0,
                g,
                dg,
                low_in,
                upp_in,
                xmin,
                xmax,
                xold1,
                xold2,
                k,
                spec,
                evaluate,
                self.multipliers,
            )
            self.multipliers = np.asarray(info["lam"], dtype=np.float64)
            self._resolves += int(info["inner"])
            if not info["conservative"]:
                self._not_conservative += 1
                for f in info["optimistic"]:
                    nm = "the objective" if f == 0 else self._row_name(int(f) - 1)
                    self._held_back[nm] = self._held_back.get(nm, 0) + 1
                if self._not_conservative == 1:
                    from jno.utils.logger import get_logger

                    get_logger().warning(
                        f"jno.optimizers.gcmma: step {k} used all max_inner={spec.max_inner} "
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
                low_in,
                upp_in,
                xmin,
                xmax,
                xold1,
                xold2,
                k,
                spec,
                spec.dual_iters,
                with_multipliers=True,
                lam0=self.multipliers,
            )
            self.multipliers = np.asarray(lam, dtype=np.float64)
        self._xold2, self._xold1 = xold1, x
        self._k = k + 1

        if not self._warned_unreachable and self.multipliers is not None and np.isnan(self.multipliers).any():
            self._warned_unreachable = True
            rows = [
                self._names[i] if i < len(self._names) and self._names[i] else f"row {i}"
                for i, bad in zip(self._ineq, np.isnan(self.multipliers))
                if bad
            ]
            from jno.utils.logger import get_logger

            # through jNO's log, like its other warnings: Python's `warnings` printed the file and a source line
            get_logger().warning(
                f"jno.optimizers.mma: at step {k} the constraint(s) {rows} cannot be met within the move limit, so "
                "their multipliers are undefined and logged as NaN; the step goes as far towards meeting them as "
                "move= allows. (Reported once; later NaN multipliers mean the same.)"
            )

        # sgd(1.0) applies `x - returned`, so hand it the negated step (in each block's own dtype).
        with jax.enable_x64(True):
            delta = x - x_new
        out = dict(grads)
        off = 0
        for lid, un, size, dt in unravels:
            out[lid] = un(jnp.asarray(delta[off : off + size], dtype=dt))
            off += size
        return out

    # docstring-from: jno.utils.adaptive.callbacks.Callback.on_training_end
    def on_training_end(self, **kw):
        """Called once after the training loop ends (also after an early stop).

        Returns:
            None.
        """
        self.report_summary()

    def _row_name(self, i):
        """The i-th inequality row's name, as the user gave it, else "row <index in the core's list>"."""
        r = self._ineq[i]
        return self._names[r] if r < len(self._names) and self._names[r] else f"row {r}"

    def report_summary(self):
        """The run's summary warning: how many GCMMA steps were accepted without the conservative guarantee."""
        if self._not_conservative:
            from jno.utils.logger import get_logger

            held = ", ".join(f"{nm} ({n})" for nm, n in sorted(self._held_back.items(), key=lambda kv: -kv[1]))
            get_logger().warning(
                f"jno.optimizers.gcmma: {self._not_conservative} of {self._k} steps ran out of "
                "max_inner re-solves and were accepted without the feasibility and descent guarantee "
                f"({self._resolves / max(self._k, 1):.1f} re-solves per step). Held back by: {held} -- the rows "
                "the approximation kept underestimating (steps). A row that changes much faster than its "
                "gradient says over one move -- a kink such as abs() at 0, a sharp smooth maximum (a small "
                "margin=), a badly scaled row -- is the usual cause: reformulate it, or lower move."
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

    def subset(self, keep):
        """Restrict the per-variable parameters to the variables the subproblem actually moves."""
        self.move = np.asarray(self.move)[keep]

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
