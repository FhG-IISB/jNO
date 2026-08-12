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
    ):
        self._physical = physical
        self._penal_lid = penal.model.layer_id
        self._rho_lid = density.model.layer_id
        self.start, self.step, self.maximum = float(start), float(step), float(maximum)
        self.tol, self.window, self.mnd_tol = float(tol), int(window), float(mnd_tol)
        self.penal = float(start)
        self.history: List[float] = []          # (epoch, penal) at each raise
        self.m_nd: Optional[float] = None
        self._losses: List[float] = []

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

        new = self.penal
        if self._converged() and self.m_nd >= self.mnd_tol and self.penal < self.maximum - 1e-12:
            new = min(self.penal + self.step, self.maximum)
            self.history.append((int(epoch), new))
            self._losses = []  # the window restarts at each continuation step (Sec. 2.4)

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
    )
