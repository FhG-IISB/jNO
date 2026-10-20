"""Augmented Lagrangian (method of multipliers) for ``jno.le`` / ``jno.ge`` -- any optimiser.

``jno.le(g, b).auglag(mu=10.0, every=100)`` (or ``(g <= b).auglag(...)``) keeps the constraint
``c = g - b <= 0`` as a reported row and adds the Powell-Hestenes-Rockafellar term

    psi(c; lam, mu) = (max(0, lam + mu c)**2 - lam**2) / (2 mu)

to the loss (Nocedal & Wright, *Numerical Optimization*, 2nd ed., 2006, Sec. 17.4, eq. 17.65).
Every ``every`` steps the multiplier takes the first-order update ``lam <- max(0, lam + mu c)``,
with ``c`` averaged over the second half of the window (so an optimiser that jitters about the
minimum does not feed its noise into the multiplier),
and ``mu`` grows tenfold whenever the violation has not fallen to a quarter of what it was at the
previous update (Bertsekas, *Constrained Optimization and Lagrange Multiplier Methods*, 1982,
Sec. 2.2). Whatever optimises the design -- optax, ``ssbfgs``, ``soap`` -- descends the augmented
loss between updates.

Where ``lam`` and ``mu`` live: in a two-entry parameter carried by the penalty term itself, whose
"optimiser" is :func:`multiplier_update`. That needs nothing from ``jno.core`` but the parameter
it already trains, because the gradient of ``psi`` with respect to ``lam`` is exactly
``max(c, -lam/mu)``: the update ``lam + mu * dpsi/dlam`` IS ``max(0, lam + mu c)``. ``jno.core``
adds the penalty rows to the loss unaveraged, which is what makes that gradient exact.

Feasibility comes only in the limit -- an iterate between updates can violate the constraint, by
O(lam/mu) early on -- which is the difference from ``mma`` / ``gcmma``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

# The violation must fall to TAU of its previous value between updates, or mu grows by GROWTH.
TAU, GROWTH, MU_MAX_FACTOR = 0.25, 10.0, 1e6


def multiplier_update(every: int, mu0: float) -> optax.GradientTransformationExtraArgs:
    """The update rule of the ``[lam, mu]`` parameter: a step every ``every`` calls, zero otherwise."""
    every = int(every)
    mu_max = float(mu0) * MU_MAX_FACTOR

    # The violation is averaged over the second half of each window rather than read at its last
    # step: an optimiser that does not settle exactly (Adam at a fixed rate jitters about the
    # minimum) otherwise feeds the update a sample of its noise, which with a large mu throws the
    # multiplier about and, never shrinking, keeps growing mu. For an optimiser that has settled,
    # the average equals the final value.
    half = every - max(1, every // 2)

    def init(params):
        z = jnp.zeros(())
        return {"count": jnp.zeros((), jnp.int32), "prev": jnp.full((), jnp.inf), "acc": z, "n": z}

    def update(grads, state, params=None, **_extra):
        (gleaf,), treedef = jax.tree_util.tree_flatten(grads)
        (pleaf,) = jax.tree_util.tree_leaves(params)
        p = pleaf.reshape(-1)
        lam, mu = p[0], p[1]
        c = gleaf.reshape(-1)[0]  # d psi / d lam = max(g, -lam/mu): the violation, complementarity-aware
        count = state["count"] + 1
        phase = (count - 1) % every  # 0 .. every-1 within the window
        take = phase >= half
        acc = state["acc"] + jnp.where(take, c, 0.0)
        n = state["n"] + jnp.where(take, 1.0, 0.0)
        due = phase == every - 1
        c_bar = acc / jnp.maximum(n, 1.0)
        lam_new = jnp.maximum(0.0, lam + mu * c_bar)
        viol = jnp.abs(c_bar)
        mu_new = jnp.where(viol > TAU * state["prev"], jnp.minimum(GROWTH * mu, mu_max), mu)
        step = jnp.where(due, jnp.stack([lam_new - lam, mu_new - mu]), jnp.zeros(2, p.dtype))
        new_state = {
            "count": count,
            "prev": jnp.where(due, viol, state["prev"]),
            "acc": jnp.where(due, 0.0, acc),
            "n": jnp.where(due, 0.0, n),
        }
        return jax.tree_util.tree_unflatten(treedef, [step.reshape(gleaf.shape).astype(gleaf.dtype)]), new_state

    return optax.GradientTransformationExtraArgs(init, update)


def penalty(residual, *, mu: float, every: int, name: str | None = None):
    """Build ``(state, psi)`` for a constraint ``residual <= 0``.

    ``state`` is the ``[lam, mu]`` parameter (``state[0]`` is the multiplier), ``psi`` the trace
    expression added to the loss.
    """
    from .. import np as jnp_ops

    mu, every = float(mu), int(every)
    if not mu > 0.0:
        raise ValueError(f"auglag: mu must be positive, got {mu}. It is the penalty weight on the violation.")
    if every < 1:
        raise ValueError(f"auglag: every must be at least 1 step between multiplier updates, got {every}.")

    state = jnp_ops.parameter((2,), name=f"auglag[{name}]" if name else "auglag")
    state.initialize(lambda key, shape, dtype=None: jnp.asarray([0.0, mu], dtype=dtype or jnp.float32))
    state.optimizer(multiplier_update(every, mu))
    lam, m = state[0], state[1]
    c = residual.mean
    psi = (jnp_ops.maximum(lam + m * c, 0.0) ** 2 - lam**2) / (2.0 * m)
    return state, psi
