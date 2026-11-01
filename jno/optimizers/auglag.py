"""Augmented Lagrangian (method of multipliers) for ``jno.le`` / ``jno.ge`` -- any optimiser.

``jno.le(g, b).auglag(mu=10.0, every=100)`` (or ``(g <= b).auglag(...)``) keeps the constraint
``c = g - b <= 0`` as a reported row and adds the Powell-Hestenes-Rockafellar term

    psi(c; lam, mu) = (max(0, lam + mu c)**2 - lam**2) / (2 mu)

to the loss (Nocedal & Wright, *Numerical Optimization*, 2nd ed., 2006, Sec. 17.4, eq. 17.65).
Every ``every`` steps the multiplier takes the first-order update ``lam <- max(0, lam + mu c)``,
with ``c`` averaged over the second half of the window (so an optimiser that jitters about the
minimum does not feed its noise into the multiplier),
and ``mu`` grows tenfold whenever the violation has not fallen to a quarter of what it was at the
previous update, unless the multiplier has already settled (Bertsekas, *Constrained Optimization and Lagrange Multiplier Methods*, 1982,
Sec. 2.2). Whatever optimises the design -- optax, ``ssbfgs``, ``soap`` -- descends the augmented
loss between updates.

Where ``lam`` and ``mu`` live: in a small parameter carried by the penalty term itself, whose
"optimiser" is :func:`multiplier_update`. That needs nothing from ``jno.core`` but the parameter
it already trains: a third entry, ``probe``, enters ``psi`` as ``(probe - stop_gradient(probe)) c``,
which is zero in value and leaves every other gradient alone, but whose own gradient is exactly
``c``. ``jno.core`` adds the penalty rows to the loss unaveraged, which is what keeps it exact.

``jno.eq(g, b).auglag()`` is the equality version: ``psi = lam c + mu c**2 / 2``, and ``lam`` is
updated without the clip at zero (Nocedal & Wright, Sec. 17.3).

Feasibility comes only in the limit -- an iterate between updates can violate the constraint, by
O(lam/mu) early on -- which is the difference from ``mma`` / ``gcmma``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

# The violation must fall to TAU of its previous value between updates, or mu grows by GROWTH.
TAU, GROWTH, MU_MAX_FACTOR = 0.25, 10.0, 1e6
# ...unless the multiplier update mu * c has fallen below this, relative to |lam| (absolute below 1).
SETTLED = 1e-6


def multiplier_update(every: int, mu0: float, equality: bool = False) -> optax.GradientTransformationExtraArgs:
    """The update rule of the ``[lam, mu, probe]`` parameter: a step every ``every`` calls, else zero.

    The gradient's third entry is the constraint value ``c`` itself (see :func:`penalty`).
    ``equality``: the multiplier of ``c == 0``, which takes either sign, so it is not clipped at zero.
    """
    every = int(every)
    mu_max = float(mu0) * MU_MAX_FACTOR

    # The constraint is averaged over the second half of each window rather than read at its last
    # step: an optimiser that does not settle exactly (Adam at a fixed rate jitters about the
    # minimum) otherwise feeds the update a sample of its noise, which with a large mu throws the
    # multiplier about and, never shrinking, keeps growing mu. For an optimiser that has settled,
    # the average equals the final value. It is the RAW value that is averaged, and the clip at
    # zero applied after: the mean of the clipped max(c, -lam/mu) is biased upwards by the jitter,
    # by more as mu grows, so it read a feasible design as violated and grew mu without end.
    half = every - max(1, every // 2)

    def init(params):
        z = jnp.zeros(())
        return {"count": jnp.zeros((), jnp.int32), "prev": jnp.full((), jnp.inf), "acc": z, "n": z}

    def update(grads, state, params=None, **_extra):
        (gleaf,), treedef = jax.tree_util.tree_flatten(grads)
        (pleaf,) = jax.tree_util.tree_leaves(params)
        p = pleaf.reshape(-1)
        lam, mu = p[0], p[1]
        c = gleaf.reshape(-1)[2]  # d psi / d probe = c, the constraint value
        count = state["count"] + 1
        phase = (count - 1) % every  # 0 .. every-1 within the window
        take = phase >= half
        acc = state["acc"] + jnp.where(take, c, 0.0)
        n = state["n"] + jnp.where(take, 1.0, 0.0)
        due = phase == every - 1
        c_bar = acc / jnp.maximum(n, 1.0)
        if equality:
            lam_new = lam + mu * c_bar
            viol, infeasible = jnp.abs(c_bar), jnp.ones((), bool)
        else:
            lam_new = jnp.maximum(0.0, lam + mu * c_bar)
            # Nocedal & Wright's measure, |max(c, -lam/mu)|: zero when feasible and complementary.
            viol, infeasible = jnp.abs(jnp.maximum(c_bar, -lam / mu)), c_bar > 0.0
        # mu grows only where it helps: an infeasible mean (for an inequality, a feasible design with
        # lam > 0 is the lam update's to correct) that has not shrunk to TAU of the last one, and a
        # multiplier that has not settled (its update below SETTLED relative to |lam|, absolute
        # below 1). Without that last condition a violation at round-off, with no reason to shrink
        # fourfold, raised mu until Adam went unstable and threw lam back to 0.
        settled = mu * viol <= SETTLED * jnp.maximum(1.0, jnp.abs(lam))
        grow = (viol > TAU * state["prev"]) & ~settled & infeasible
        mu_new = jnp.where(grow, jnp.minimum(GROWTH * mu, mu_max), mu)
        step = jnp.where(due, jnp.stack([lam_new - lam, mu_new - mu, 0.0 * lam]), jnp.zeros(3, p.dtype))
        new_state = {
            "count": count,
            "prev": jnp.where(due, viol, state["prev"]),
            "acc": jnp.where(due, 0.0, acc),
            "n": jnp.where(due, 0.0, n),
        }
        return jax.tree_util.tree_unflatten(treedef, [step.reshape(gleaf.shape).astype(gleaf.dtype)]), new_state

    return optax.GradientTransformationExtraArgs(init, update)


def penalty(residual, *, mu: float, every: int, name: str | None = None, equality: bool = False):
    """Build ``(state, psi)`` for a constraint ``residual <= 0``, or ``residual == 0`` with ``equality``.

    The equality term is ``lam c + mu c**2 / 2`` (Nocedal & Wright, Sec. 17.3, with the multiplier's
    sign flipped so that both forms update as ``lam + mu c``). Its gradient with respect to ``lam``
    is ``c``.

    ``state`` is the ``[lam, mu, probe]`` parameter (``state[0]`` is the multiplier; ``probe`` is
    always 0 and only carries ``c`` to the update as its gradient), ``psi`` the trace expression
    added to the loss.
    """
    from .. import np as jnp_ops
    from ..jnp_ops import _unary_call

    mu, every = float(mu), int(every)
    if not mu > 0.0:
        raise ValueError(f"auglag: mu must be positive, got {mu}. It is the penalty weight on the violation.")
    if every < 1:
        raise ValueError(f"auglag: every must be at least 1 step between multiplier updates, got {every}.")

    state = jnp_ops.parameter((3,), name=f"auglag[{name}]" if name else "auglag")
    state.initialize(lambda key, shape, dtype=None: jnp.asarray([0.0, mu, 0.0], dtype=dtype or jnp.float32))
    state.optimizer(multiplier_update(every, mu, equality=equality))
    lam, m, probe = state[0], state[1], state[2]
    c = residual.mean
    if equality:
        psi = lam * c + 0.5 * m * c**2
    else:
        psi = (jnp_ops.maximum(lam + m * c, 0.0) ** 2 - lam**2) / (2.0 * m)
    # Zero in value and in every gradient but its own, which is c: the multiplier update reads the
    # constraint from there, unclipped, whatever the sign of lam + mu c.
    psi = psi + (probe - _unary_call(jax.lax.stop_gradient, probe)) * c
    return state, psi
