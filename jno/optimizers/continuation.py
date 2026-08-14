"""SIMP penalisation continuation — raise ``penal`` once the objective has settled.

Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403, Sec. 2.3.2 and Fig. 4a. Solving
straight at a high penalisation is a well-known trap: the problem is strongly non-convex and the
optimiser lands in a poor local minimum. So SIMP starts at ``penal = 3``, which establishes the
topology while leaving intermediate ("grey") density behind, and only then raises the exponent to
squeeze that grey out — the topology barely moves, the design goes black-and-white.

The trigger is the paper's, and it is two conditions rather than a fixed schedule:

    the objective has converged  AND  the grey-level indicator is still above its tolerance
        -> penal += 1, and the convergence window restarts

with convergence meaning the relative change in the objective is below ``tol`` across ``window``
consecutive logged intervals, and (eq. 20)

    M_nd = 4 * mean( rho_bar * (1 - rho_bar) )

measured on the **physical** density — after any ``constrain(...)`` reparameterisation, since that
is the field the stiffness actually sees.

``penal`` is an ordinary ``jno.np.parameter`` used as the exponent in the weak form, so raising it
needs no recompilation. It is written the way ENGD and MMA write their steps: the callback returns
``penal - penal_new`` as its "gradient" and the injected ``optax.sgd(1.0)`` lands the new value
exactly. Its own loss gradient never applies, because this hook replaces that entry every step.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ..utils.adaptive.callbacks import Callback as _Callback


class SIMPContinuation(_Callback):
    """Raise the SIMP exponent on the paper's schedule. See :func:`simp_continuation`."""

    def __init__(
        self,
        penal: Any,
        density: Any,
        *,
        start: float = 3.0,
        step: float = 1.0,
        maximum: float = 8.0,
        tol: float = 1e-4,
        window: int = 3,
        mnd_tol: float = 1e-2,
        physical=None,
        watch: Optional[int] = None,
        every: int = 1,
        patience: Optional[int] = None,
    ):
        self._watch = watch
        self.every = max(int(every), 1)
        self._physical = physical
        self._penal_lid = penal.model.layer_id
        self._rho_lid = density.model.layer_id
        self.start, self.step, self.maximum = float(start), float(step), float(maximum)
        self.tol, self.window, self.mnd_tol = float(tol), int(window), float(mnd_tol)
        self.patience = None if patience is None else max(int(patience), 1)
        self.penal = float(start)
        self.history: List[float] = []          # (epoch, penal) at each raise
        self.m_nd: Optional[float] = None
        self._losses: List[float] = []
        self._waited = 0                        # samples since the last raise, for `patience`

    def _converged(self) -> bool:
        """Relative change below ``tol`` across ``window`` consecutive steps."""
        if len(self._losses) < self.window + 1:
            return False
        recent = self._losses[-(self.window + 1) :]
        return all(
            abs(b - a) <= self.tol * max(abs(a), 1e-30) for a, b in zip(recent[:-1], recent[1:])
        )

    def on_before_update(self, *, grads, trainable, epoch, **kw):
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        # The convergence window samples every `every` iterations, so it spans an INTERVAL rather
        # than a single step -- which is what the paper's criterion is ("the relative changes in
        # the objective function over three consecutive iteration intervals"). Per-iteration
        # sampling is far stricter: an optimiser still making steady progress trips it never, and
        # `penal` then sits at its initial value for the whole run.
        sample = int(epoch) % self.every == 0
        # NOTE: returning early here would be a bug, not an optimisation. This hook OWNS the
        # `penal` entry: on any step where it does not overwrite it, the raw loss gradient w.r.t.
        # `penal` survives and `sgd(1.0)` applies it, so `penal` random-walks and the run diverges
        # (measured: compliance 79 -> 3.7e10 with `every=25`). Only the bookkeeping is strided.

        # Which quantity has to settle. `total_loss` is the whole objective, which under a
        # perimeter barrier is C - beta R -- and that moves for as long as beta decays, so the
        # window would never close. `watch=i` follows `individual_losses[i]` instead, i.e. the
        # compliance alone, which is what Fig. 4a's "C converged?" actually asks.
        if sample:
            if self._watch is not None:
                indiv = kw.get("individual_losses")
                if indiv is not None:
                    self._losses.append(float(np.asarray(indiv).reshape(-1)[self._watch]))
            else:
                total = kw.get("total_loss")
                if total is not None:
                    self._losses.append(float(total))

        # M_nd on the PHYSICAL density. `trainable` is the partitioned half, so a paramax
        # `constrain(...)` wrapper is NOT applied here -- its function lives in the static half and
        # unwrapping raises. Pass the same map as `physical=` and it is applied explicitly;
        # measuring the raw design density instead would call a design binary while the field the
        # stiffness actually sees is still grey.
        leaves = [
            lf for lf in jax.tree_util.tree_leaves(trainable[self._rho_lid]) if eqx.is_inexact_array(lf)
        ]
        r = jnp.concatenate([lf.reshape(-1) for lf in leaves])
        if self._physical is not None:
            r = jnp.asarray(self._physical(r)).reshape(-1)
        self.m_nd = float(4.0 * jnp.mean(r * (1.0 - r)))

        # The convergence gate assumes the objective eventually settles. When it does not -- an
        # unregularised run that fragments instead of converging, or an MMA iterate that keeps
        # oscillating above `tol` -- `penal` stays at `start` for the whole run and the design has
        # no reason to go binary. That is backwards: penalisation is needed MOST where convergence
        # is worst. `patience` raises anyway after that many samples without a raise, so a stalled
        # run still gets its continuation. Left as None the behaviour is exactly the paper's.
        if sample:
            self._waited += 1
        stalled = self.patience is not None and self._waited >= self.patience

        new = self.penal
        if (
            sample
            and (self._converged() or stalled)
            and self.m_nd >= self.mnd_tol
            and self.penal < self.maximum - 1e-12
        ):
            new = min(self.penal + self.step, self.maximum)
            self.history.append((int(epoch), new, "stalled" if stalled and not self._converged() else "converged"))
            self._losses = []  # the window restarts at each continuation step (Sec. 2.4)
            self._waited = 0

        delta = self.penal - new
        self.penal = new
        # `sgd(1.0)` applies `x - returned`, so hand it the negated step -- ENGD and MMA both
        # abuse the return value this way, and zero when nothing changes leaves penal alone.
        grads = dict(grads)
        grads[self._penal_lid] = jax.tree_util.tree_map(
            lambda x: jnp.full_like(x, delta), grads[self._penal_lid]
        )
        return grads


def simp_continuation(
    penal: Any,
    density: Any,
    *,
    start: float = 3.0,
    step: float = 1.0,
    maximum: float = 8.0,
    tol: float = 1e-4,
    window: int = 3,
    mnd_tol: float = 1e-2,
    physical=None,
    watch: Optional[int] = None,
    every: int = 1,
    patience: Optional[int] = None,
) -> SIMPContinuation:
    """Raise the SIMP exponent once the objective settles and the design is still grey.

    Register it **after** the design optimiser's own callback -- ``jno.core`` applies
    ``on_before_update`` hooks in list order, each seeing the previous one's gradients, and this
    one only replaces the ``penal`` entry.

    Args:
        penal: The ``jno.np.parameter`` used as the SIMP exponent in the weak form. Give it
            ``optax.sgd(1.0)`` and initialise it to ``start``; this hook writes it directly.
        density: The design-density parameter, read to compute ``M_nd`` (eq. 20).
        start: Initial exponent. 3 is the paper's, and the field's, default.
        step: Increment per continuation step.
        maximum: Ceiling, so a design that never binarises cannot run the exponent away.
        tol: Relative-change tolerance for "the objective has converged" (the paper's 1e-4).
        window: How many consecutive intervals must all be below ``tol`` (the paper's three).
        mnd_tol: Grey-level target. Above this, the exponent keeps rising (the paper's 1e-2).
        physical: Optional map from the design density to the physical one, e.g.
            ``d.patch_filter()``. Pass it whenever the density is reparameterised with
            ``constrain(...)``: the hook sees the partitioned trainable half, where the paramax
            wrapper cannot be applied, so ``M_nd`` would otherwise be measured on the wrong field.
        watch: Index into ``individual_losses`` whose convergence is tested, instead of the total
            loss. Pass the compliance term's index whenever the objective also carries a decaying
            penalty -- the total then keeps moving because the penalty does, and the window never
            closes.
        every: Sample the convergence window every ``n`` iterations, so ``window`` samples span an
            *interval* rather than ``window`` consecutive steps. The paper's criterion is over
            three iteration intervals; with ``every=1`` the test is much stricter than theirs and
            a run that is still making steady progress never satisfies it.

    Example::

        penal = jno.np.parameter((1,), name="penal")
        penal.initialize(lambda k, s, dtype=None: jnp.full(s, 3.0))
        penal.optimizer(optax.sgd(1.0))
        cont = jno.optimizers.simp_continuation(penal, rho)
        jno.core([...], domain=d).solve(300, callbacks=[cont])
        print(cont.history)   # [(epoch, penal), ...] -- when each step fired

    References:
        Jung, Yun & Kim, *Computers & Structures* **331** (2026) 108403, Sec. 2.3.2, Fig. 4a.
        Sigmund, *Struct. Multidisc. Optim.* **21**(2), 2001, 120-127 (SIMP and continuation).
    """
    return SIMPContinuation(
        penal,
        density,
        start=start,
        step=step,
        maximum=maximum,
        tol=tol,
        window=window,
        mnd_tol=mnd_tol,
        physical=physical,
        watch=watch,
        every=every,
        patience=patience,
    )


class GeometricDecay(_Callback):
    """Multiply a scalar parameter by ``gamma`` every iteration. See :func:`geometric_decay`."""

    def __init__(self, param: Any, gamma: float, *, start: float = 1.0, minimum: float = 0.0):
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError(f"geometric_decay: gamma must be in (0, 1], got {gamma!r}.")
        self._lid = param.model.layer_id
        self.gamma, self.minimum = float(gamma), float(minimum)
        self.value = float(start)
        self.history: List[float] = []

    def on_before_update(self, *, grads, epoch, **kw):
        import jax
        import jax.numpy as jnp

        new = max(self.value * self.gamma, self.minimum)
        delta = self.value - new
        self.value = new
        self.history.append(new)
        grads = dict(grads)
        grads[self._lid] = jax.tree_util.tree_map(lambda x: jnp.full_like(x, delta), grads[self._lid])
        return grads


def geometric_decay(param: Any, gamma: float, *, start: float = 1.0, minimum: float = 0.0) -> GeometricDecay:
    """Decay a scalar parameter geometrically, ``b_iter = gamma * b_(iter-1)`` — eq. (41).

    Written for the perimeter barrier's weight (Jung, Yun & Kim, *Computers & Structures* **331**
    (2026) 108403, eq. 40-41). The objective is ``C - beta * P* log(P* - P)``: with ``beta`` fixed,
    the barrier holds the design well short of the target ``P*`` forever, because the penalty is
    still paying for perimeter it is allowed to spend. Shrinking ``beta`` lets the design approach
    ``P*`` from below as the run proceeds — the paper's own words, "to reduce the approximation
    error of the perimeter constraint", and the standard interior-point continuation.

    Give the parameter ``optax.sgd(1.0)`` and initialise it to ``start``; this hook writes it
    directly, the same way :class:`SIMPContinuation` writes ``penal``.

    Args:
        param: The scalar ``jno.np.parameter`` to decay.
        gamma: Decay factor per iteration, in ``(0, 1]``. ``1.0`` is no decay.
        start: Initial value. Must match the parameter's initializer.
        minimum: Floor, so the barrier never vanishes entirely and let the design cross ``P*``.
    """
    return GeometricDecay(param, gamma, start=start, minimum=minimum)
