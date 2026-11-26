"""DSGNAR: Doubly-Sketched Gauss-Newton with Adaptive Ratio -- a Levenberg-Marquardt step computed
on a small square sketch of the residual Jacobian, its length set by probing for a target gain ratio.

Based on:
  Webb, J. J., Jerad, S., & Cartis, C. (2026).
  An Optimisation Framework for the Well-Conditioned Training of Physics-Informed Neural Networks.
  arXiv:2607.02194.
Implemented from the paper's text (Secs. 3.2-3.4, Algorithms 1, 3 and 5); no code was used.

**Sketch** (Sec. 3.2, drawn once per ``solve()`` and kept, Sec. 3.2.2). With the weighted residual
:math:`\\hat r \\in \\mathbb{R}^m` (rows scaled by :math:`\\sqrt{w_k/N_k}`, Eqs. (9)-(10)) and its Jacobian
:math:`J`, the method works with

.. math::

    \\tilde J = C J \\Omega S \\in \\mathbb{R}^{s \\times s}, \\qquad \\tilde r = C \\hat r \\in \\mathbb{R}^s,

:math:`C` a CountSketch with :math:`K` hashes (Eq. (16)) and :math:`\\Omega S = D\\Pi F S` a subsampled
randomized cosine transform (Eq. (17): signs, permutation, unitary DCT-II, restriction to ``s``
coordinates). :math:`J\\Omega S` is ``s`` Jacobian-vector products; :math:`J` itself is never formed.

**Step** (Sec. 3.3). One SVD :math:`\\tilde J = U\\Sigma V^\\top` gives every damped step (Eq. (20))

.. math::

    \\tilde p(\\lambda) = -V \\operatorname{diag}\\!\\big(\\sigma_i / (\\sigma_i^2 + \\lambda)\\big) U^\\top \\tilde r,
    \\qquad p = \\Omega S\\, \\tilde p,

and :math:`\\lambda` for a radius :math:`\\delta` solves :math:`\\|\\tilde p(\\lambda)\\| = \\delta` (the secular
equation (21), solved here by bisection in :math:`\\log\\lambda`). The gain ratio is
:math:`\\varrho = (L(\\theta) - L(\\theta + p)) / (\\tilde m(0) - \\tilde m(\\tilde p))`, actual decrease of
:math:`L = \\tfrac12\\|\\hat r\\|^2` over the decrease of the sketched model
:math:`\\tilde m(\\tilde p) = \\tfrac12\\|\\tilde r + \\tilde J\\tilde p\\|^2`.

**Adaptive ratio** (Algorithm 3). 24 radii, geometric between :math:`\\Delta/3` and :math:`3\\Delta`, are
probed; the ratios are made non-increasing in the radius and interpolated to the target
:math:`\\varrho^\\star`, which picks the radius (and so :math:`\\lambda`) of the step; outside the probed
range the nearest probe is used. A step that lowers :math:`L` with :math:`\\varrho > 0` is accepted and
its radius kept as :math:`\\Delta`; otherwise the parameters stay and :math:`\\Delta \\leftarrow \\Delta/3`,
floored at :math:`10^{-8}` (float64) or :math:`10^{-4}` (float32). The target starts at the conservative
Stage 1 value (0.075 in float64, 0.1 in float32) and switches for good to Stage 2 (0.5) once a line
fitted to the last 30 values of :math:`\\log\\lambda` has slope :math:`> 10^{-4}` and correlation
:math:`> 0.1` (Algorithm 5): :math:`\\lambda` has passed its minimum.

Departures from the paper, made here: the ratio is interpolated linearly in :math:`\\log\\delta`
between the two probes around the target, not with PCHIP; a step that overshoots the target ratio
is not rejected (the paper names no margin for it); the condition weights stay those of
``gram_terms`` (the paper's weight update, Algorithm 4, is not applied); every iteration runs in the
working precision (the paper runs its first ten double-precision iterations in single precision);
the Jacobian sketch is accumulated in one pass, not in batches of rows.

**Not yet validated against the paper's numbers.** The paper's smallest case has a 700 x 700 sketch.
On a 129-parameter Poisson PINN (a 43 x 43 sketch of 1020 rows) the sketched gradient
:math:`\\Omega S \\tilde J^\\top \\tilde r` had cosine between -0.4 and +0.4 with the true gradient over
five sketches, so with one fixed sketch the small trial steps often point uphill and are rejected:
from a 0.1-scaled initialisation no step was accepted in 200 epochs, from a standard one the
error fell from about 1 to 3.4e-2 (Levenberg-Marquardt: 4.9e-5). Prefer :func:`lm`, :func:`dngd` or
:func:`nystrom_ngd` until this is checked at the paper's scale.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.fft import dct

from ._residual_callback import half_sq_norm
from ._residuals import JACOBIAN_CHUNK
from ._sentinel import ResidualOptimizer

_N_PROBES = 24
_WINDOW = 30
_SLOPE_TOL, _CORR_TOL = 1e-4, 0.1


class _Sketch(NamedTuple):
    rows: jax.Array  # (K, m) CountSketch target row of each residual row
    signs: jax.Array  # (K, m) +-1
    flip: jax.Array  # (P,) D
    perm: jax.Array  # (P,) Pi
    keep: jax.Array  # (s,) S: the kept coordinates


class _DSGNARState(NamedTuple):
    sketch: _Sketch
    radius: jax.Array
    stage2: jax.Array  # bool
    log_lams: jax.Array  # (W,) ring buffer of log(lambda) of accepted steps
    n_lams: jax.Array  # int, how many have been written


def _draw_sketch(key, m: int, P: int, s: int, hashes: int, dtype) -> _Sketch:
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    rows = jax.random.randint(k1, (hashes, m), 0, s)
    signs = jax.random.rademacher(k2, (hashes, m), dtype=dtype)
    flip = jax.random.rademacher(k3, (P,), dtype=dtype)
    perm = jax.random.permutation(k4, P)
    keep = jax.random.choice(k5, P, (s,), replace=False)
    return _Sketch(rows, signs, flip, perm, keep)


def _count_sketch(sk: _Sketch, v, s: int):
    """``C v`` for a length-``m`` vector (Eq. (16))."""
    K = sk.rows.shape[0]
    vals = (sk.signs * v[None, :]).reshape(-1)
    return jax.ops.segment_sum(vals, sk.rows.reshape(-1), num_segments=s) / jnp.sqrt(K)


def _lift(sk: _Sketch, pt, P: int):
    """``Ω S p̃ = D Π F S p̃`` (Eq. (17)) for a length-``s`` vector."""
    x = jnp.zeros((P,), pt.dtype).at[sk.keep].set(pt)
    return sk.flip * dct(x, type=2, norm="ortho")[sk.perm]


def _lm_lambda_for_radius(sig, b, delta, iters: int = 80):
    """``λ >= 0`` with ``||p̃(λ)|| = δ`` (bisection in log λ); ``0`` when the undamped step is shorter."""
    tiny = jnp.finfo(sig.dtype).tiny
    smax = jnp.max(sig)
    keep = sig > jnp.finfo(sig.dtype).eps * smax * sig.shape[0]

    def norm(lam):
        coef = jnp.where(keep, sig / (sig**2 + lam), 0.0)
        return jnp.linalg.norm(coef * b)

    lo = jnp.log(jnp.maximum(jnp.finfo(sig.dtype).eps * smax**2, tiny))
    hi = jnp.log(jnp.maximum(smax * jnp.linalg.norm(b) / jnp.maximum(delta, tiny), tiny)) + 1.0

    def body(_, c):
        lo, hi = c
        mid = 0.5 * (lo + hi)
        too_long = norm(jnp.exp(mid)) > delta
        return jnp.where(too_long, mid, lo), jnp.where(too_long, hi, mid)

    lo, hi = jax.lax.fori_loop(0, iters, body, (lo, hi))
    return jnp.where(norm(0.0) <= delta, 0.0, jnp.exp(hi))


def _stage_switch(log_lams, n):
    """Algorithm 5: the slope and the Pearson correlation of log λ against the iteration index."""
    W = log_lams.shape[0]
    # the ring buffer holds the last W values; order them oldest first
    idx = (jnp.arange(W) + n) % W
    y = log_lams[idx]
    x = jnp.arange(W, dtype=y.dtype)
    xc, yc = x - x.mean(), y - y.mean()
    slope = jnp.vdot(xc, yc) / jnp.vdot(xc, xc)
    corr = jnp.vdot(xc, yc) / jnp.sqrt(jnp.maximum(jnp.vdot(xc, xc) * jnp.vdot(yc, yc), jnp.finfo(y.dtype).tiny))
    return (n >= W) & (slope > _SLOPE_TOL) & (corr > _CORR_TOL)


def _dsgnar_direction(r_of, flat_p, state: _DSGNARState, *, s: int):
    """One DSGNAR step; returns ``(d, state, info)`` with new parameters ``θ - d``."""
    sk = state.sketch
    P = flat_p.shape[0]
    r, jvp = jax.linearize(r_of, flat_p)
    dtype = r.dtype
    f0 = half_sq_norm(r)
    rt = _count_sketch(sk, r, s)

    def sketched_col(i):
        return _count_sketch(sk, jvp(_lift(sk, jax.nn.one_hot(i, s, dtype=dtype), P)), s)

    Jt = jax.lax.map(sketched_col, jnp.arange(s), batch_size=min(JACOBIAN_CHUNK, s)).T  # (s, s)
    U, sig, Vt = jnp.linalg.svd(Jt, full_matrices=False)
    b = U.T @ rt
    keep = sig > jnp.finfo(dtype).eps * jnp.max(sig) * s

    def step_for(lam):
        coef = jnp.where(keep, sig / (sig**2 + lam), 0.0)
        pt = -(Vt.T @ (coef * b))
        res = rt + Jt @ pt
        pred = 0.5 * jnp.vdot(rt, rt) - 0.5 * jnp.vdot(res, res)
        return pt, pred

    def probe(delta):
        lam = _lm_lambda_for_radius(sig, b, delta)
        pt, pred = step_for(lam)
        p = _lift(sk, pt, P)
        f1 = half_sq_norm(r_of(flat_p + p))
        rho = (f0 - f1) / pred
        return jnp.where(jnp.isfinite(rho) & (pred > 0), rho, -jnp.inf)

    radius = state.radius
    deltas = radius * jnp.exp(jnp.linspace(jnp.log(1.0 / 3.0), jnp.log(3.0), _N_PROBES, dtype=dtype))
    rhos = jax.lax.map(probe, deltas)
    rhos_mono = jax.lax.cummin(rhos)  # non-increasing in the radius (Algorithm 3, step 2)

    double = jnp.finfo(dtype).bits >= 64
    target = jnp.where(state.stage2, 0.5, jnp.where(double, 0.075, 0.1))
    above = rhos_mono >= target
    n_above = jnp.sum(above)
    # last probe still at or above the target, and the next one below it: interpolate in log(delta)
    i = jnp.clip(n_above - 1, 0, _N_PROBES - 2)
    l0, l1 = jnp.log(deltas[i]), jnp.log(deltas[i + 1])
    r0, r1 = rhos_mono[i], rhos_mono[i + 1]
    w = jnp.clip((r0 - target) / jnp.where(r0 - r1 > 0, r0 - r1, 1.0), 0.0, 1.0)
    w = jnp.where(jnp.isfinite(w), w, 0.0)
    log_delta = jnp.where(
        n_above == 0, jnp.log(deltas[0]), jnp.where(n_above == _N_PROBES, jnp.log(deltas[-1]), l0 + w * (l1 - l0))
    )
    delta_star = jnp.exp(log_delta)

    lam = _lm_lambda_for_radius(sig, b, delta_star)
    pt, pred = step_for(lam)
    p = _lift(sk, pt, P)
    f1 = half_sq_norm(r_of(flat_p + p))
    rho = (f0 - f1) / pred
    accept = jnp.isfinite(f1) & (f1 < f0) & (pred > 0) & (rho > 0)

    floor = jnp.where(double, 1e-8, 1e-4).astype(dtype)
    new_radius = jnp.where(accept, delta_star, jnp.maximum(radius / 3.0, floor))
    W = state.log_lams.shape[0]
    log_lams = jnp.where(
        accept, state.log_lams.at[state.n_lams % W].set(jnp.log(jnp.maximum(lam, jnp.finfo(dtype).tiny))), state.log_lams
    )
    n_lams = state.n_lams + accept.astype(state.n_lams.dtype)
    stage2 = state.stage2 | (accept & _stage_switch(log_lams, n_lams))
    new_state = _DSGNARState(sk, new_radius, stage2, log_lams, n_lams)
    info = {
        "radius": delta_star,
        "damping": lam,
        "rho": rho,
        "accepted": accept,
        "stage2": stage2,
        "loss": f0,
        "target": target,
    }
    return jnp.where(accept, -p, jnp.zeros_like(p)), new_state, info


def _default_sketch_size(P: int, m: int) -> int:
    return max(1, min(P // 3, P, m))


# docstring-from: jno.optimizers.dsgnar.dsgnar
class DSGNAROptimizer(ResidualOptimizer):
    r"""Doubly-Sketched Gauss-Newton with Adaptive Ratio: sketched Levenberg-Marquardt steps whose length
    is set by probing for a target gain ratio.

    Pass to ``net.optimizer(...)``. Each epoch pushes ``sketch_size`` Jacobian-vector products
    through the residual, sketches them to an ``s x s`` matrix, takes its SVD, and probes 24 trial
    radii (24 residual evaluations) to choose the step; a step that does not lower the loss is
    rejected and the radius cut by 3. The damping is never set by hand: it follows from the chosen
    radius. Use float64; the paper reaches rel-L2 near machine precision on smooth problems.

    Args:
        gram_terms: ``None`` (default): one ``(residual.grad(model), 1.0)`` pair per loss term that
            involves the model. Otherwise explicit ``(NetworkGradient, weight)`` pairs, whose weights
            are the paper's condition weights :math:`w_m`.
        sketch_size: ``s``. ``None`` (default) uses :math:`\lfloor P/3 \rfloor`, the low end of the
            paper's range :math:`[P/3, P/2]`; capped at the parameter count and the residual rows.
        hashes: CountSketch hashes :math:`K` per row (default 2; the paper calls 2 or 4 typical).
        radius: Initial trust radius :math:`\Delta_0` (default 1, the paper's value).
        seed: Seed of the sketch, drawn once per ``solve()``.

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(jno.optimizers.dsgnar())
        jno.core([pde.mse, bc.mse]).solve(300)

    References:
        Webb, Jerad & Cartis, *An Optimisation Framework for the Well-Conditioned Training of
        Physics-Informed Neural Networks*, 2026, arXiv:2607.02194, Eqs. (16)-(21), Algorithms 1, 3, 5.
    """

    _who = "jno.optimizers.dsgnar"

    def __init__(self, gram_terms=None, *, sketch_size=None, hashes: int = 2, radius: float = 1.0, seed: int = 0):
        if sketch_size is not None and int(sketch_size) < 1:
            raise ValueError(f"jno.optimizers.dsgnar: sketch_size must be >= 1, got {sketch_size!r}.")
        if int(hashes) < 1:
            raise ValueError(f"jno.optimizers.dsgnar: hashes must be >= 1, got {hashes!r}.")
        if not float(radius) > 0.0:
            raise ValueError(f"jno.optimizers.dsgnar: radius must be > 0, got {radius!r}.")
        super().__init__(gram_terms)
        self._sketch_size, self._hashes, self._radius, self._seed = sketch_size, int(hashes), float(radius), int(seed)

    def build_callback(self, gram_terms):
        """Return the callback that computes this optimizer's step inside ``solve()``.

        Args:
            gram_terms: ``(NetworkGradient, weight)`` pairs, one per residual class (auto-detected by
                ``solve()`` unless given to the factory).
        """
        from ._residual_callback import ResidualStepCallback

        spec = self

        class DSGNARCallback(ResidualStepCallback):
            who = "jno.optimizers.dsgnar"

            def _s(self):
                s = spec._sketch_size or _default_sketch_size(self.P, self.m)
                return int(min(s, self.P, self.m))

            def init_state(self, flat_p, m):
                dtype = flat_p.dtype
                sk = _draw_sketch(jax.random.PRNGKey(spec._seed), m, self.P, self._s(), spec._hashes, dtype)
                return _DSGNARState(
                    sk, jnp.asarray(spec._radius, dtype), jnp.asarray(False), jnp.zeros(_WINDOW, dtype), jnp.asarray(0)
                )

            def direction(self, r_of, flat_p, state):
                return _dsgnar_direction(r_of, flat_p, state, s=self._s())

        return DSGNARCallback(gram_terms)

    def __repr__(self):
        return f"DSGNAROptimizer(sketch_size={self._sketch_size}, hashes={self._hashes}, radius={self._radius})"


def dsgnar(gram_terms=None, *, sketch_size=None, hashes: int = 2, radius: float = 1.0, seed: int = 0) -> DSGNAROptimizer:
    """Doubly-Sketched Gauss-Newton with Adaptive Ratio: sketched Levenberg-Marquardt steps whose length
    is set by probing for a target gain ratio.

    Pass to ``net.optimizer(...)``. Each epoch pushes ``sketch_size`` Jacobian-vector products
    through the residual, sketches them to an ``s x s`` matrix, takes its SVD, and probes 24 trial
    radii (24 residual evaluations) to choose the step; a step that does not lower the loss is
    rejected and the radius cut by 3. The damping is never set by hand: it follows from the chosen
    radius. Use float64; the paper reaches rel-L2 near machine precision on smooth problems.

    Args:
        gram_terms: ``None`` (default): one ``(residual.grad(model), 1.0)`` pair per loss term that
            involves the model. Otherwise explicit ``(NetworkGradient, weight)`` pairs, whose weights
            are the paper's condition weights :math:`w_m`.
        sketch_size: ``s``. ``None`` (default) uses :math:`\\lfloor P/3 \\rfloor`, the low end of the
            paper's range :math:`[P/3, P/2]`; capped at the parameter count and the residual rows.
        hashes: CountSketch hashes :math:`K` per row (default 2; the paper calls 2 or 4 typical).
        radius: Initial trust radius :math:`\\Delta_0` (default 1, the paper's value).
        seed: Seed of the sketch, drawn once per ``solve()``.

    Example::

        jax.config.update("jax_enable_x64", True)
        net.optimizer(jno.optimizers.dsgnar())
        jno.core([pde.mse, bc.mse]).solve(300)

    References:
        Webb, Jerad & Cartis, *An Optimisation Framework for the Well-Conditioned Training of
        Physics-Informed Neural Networks*, 2026, arXiv:2607.02194, Eqs. (16)-(21), Algorithms 1, 3, 5.
    """
    return DSGNAROptimizer(gram_terms, sketch_size=sketch_size, hashes=hashes, radius=radius, seed=seed)
