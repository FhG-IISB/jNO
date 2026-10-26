"""``jno.optimizers.bounded(opt, lower=, upper=)`` -- box bounds for any optax optimiser.

After every step of ``opt`` the parameter is projected back onto ``[lower, upper]``, so the update
handed to the parameter is ``clip(x + u, lower, upper) - x``. For a first-order optimiser this is
the projected gradient method (Bertsekas, *Nonlinear Programming*, 3rd ed., 2016, Sec. 3.3). A
quasi-Newton optimiser (``ssbfgs``) still runs, but its curvature pairs then come from projected
steps, so expect it to lose its fast convergence once bounds are active.

It is a sentinel rather than a ``GradientTransformation`` because ``jno.core`` chains a
learning-rate ``optax.scale`` AFTER the user's optimiser; the projection has to come after that,
or a ``.scale()`` would carry the step out of the box again.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax


class BoundedOptimizer:
    """An optimiser with its iterates projected onto ``[lower, upper]``: what :func:`bounded` returns.

    Each step of ``inner`` is followed by a projection (clipping) of the parameter onto the box, after
    jNO's learning-rate scale has been applied.

    Args:
        inner: the optimiser being bounded (an optax transform, or one of jNO's sentinels).
        lower: lower bound, a scalar or an array broadcasting to the parameter's shape.
        upper: upper bound, likewise.
    """

    def __init__(self, inner, lower, upper):
        from .engd import ENGDOptimizer
        from .md_decouple import MDOptimizer
        from .mma import MMAOptimizer

        if isinstance(inner, (MMAOptimizer, ENGDOptimizer, MDOptimizer, BoundedOptimizer)):
            raise TypeError(
                f"jno.optimizers.bounded wraps an optax optimiser, not {type(inner).__name__}. "
                "mma/gcmma take lower=/upper= themselves."
            )
        lo, hi = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
        if np.any(lo > hi):
            raise ValueError("jno.optimizers.bounded: lower must not exceed upper anywhere.")
        if not (np.any(np.isfinite(lo)) or np.any(np.isfinite(hi))):
            raise ValueError("jno.optimizers.bounded: give lower=, upper= or both; with neither it does nothing.")
        self.inner, self.lower, self.upper = inner, lower, upper

    def project(self, chain: optax.GradientTransformation) -> optax.GradientTransformation:
        """``chain`` (the optimiser as ``jno.core`` built it) followed by the projection."""
        lower, upper = self.lower, self.upper

        def update(updates, state, params=None, **extra):
            if params is None:
                raise ValueError("jno.optimizers.bounded needs the parameters to project onto the box.")
            u, state = chain.update(updates, state, params, **extra)

            def step(p, d):
                lo, hi = jnp.asarray(lower, p.dtype), jnp.asarray(upper, p.dtype)
                return (jnp.clip(p + d, lo, hi) - p).astype(d.dtype)

            return jax.tree.map(step, params, u), state

        return optax.GradientTransformationExtraArgs(chain.init, update)


def bounded(opt, *, lower=-np.inf, upper=np.inf) -> BoundedOptimizer:
    """Keep a parameter inside ``[lower, upper]`` while ``opt`` optimises it.

    ``lower`` / ``upper`` are scalars or arrays that broadcast to the parameter's shape; either may
    be omitted (an infinite bound). A start outside the box is projected onto it by the first step::

        rho.optimizer(jno.optimizers.bounded(optax.adam(1e-2), lower=0.0, upper=1.0))

    With ``jno.le(...).auglag()`` this lets any first-order optimiser handle a bounded, constrained
    design. ``mma`` / ``gcmma`` take bounds themselves and are not wrapped.
    """
    return BoundedOptimizer(opt, lower, upper)
