"""``jno.solve.etdrk4()``: fourth-order exponential time differencing on an all-Fourier spectral problem.

For ``m u̇ + L u + N(u, t) = 0`` with ``L`` a constant-coefficient linear operator -- diagonal in Fourier space, its
symbol read from the equation's own terms -- the linear part is integrated exactly and the rest by the
fourth-order Runge-Kutta-like stages of Cox & Matthews, with the φ-functions evaluated by a contour integral, which
is stable where their Taylor and closed forms cancel (Kassam & Trefethen). The state lives in coefficient space for
the whole march: a stage costs the transforms of the nonlinear terms only.

Algebraic unknowns (an equation without ``u.t``, e.g. ``Δψ + ω = 0``) are condensed at every stage: their equation
must be linear with constant coefficients in the unknown itself (``S ψ̂ = -r̂(ω)``), solved mode by mode; a mode
with a zero symbol takes the value the unknown's gauge row asks for (``psi.mean``, a pin).

Based on:
  Cox, S. M. & Matthews, P. C. (2002). Exponential time differencing for stiff systems. J. Comput. Phys. 176,
  430-455.
  Kassam, A.-K. & Trefethen, L. N. (2005). Fourth-order time-stepping for stiff PDEs. SIAM J. Sci. Comput. 26(4),
  1214-1233.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .._strong_form import _contains_unknown, _has_temporal_of, _unwrap
from . import fourier_nd as fnd
from .trace import derivative_chain

__all__ = ["ETDRK4Scheme", "etd_march", "split_terms"]


class ETDRK4Scheme:
    """The ETDRK4 time scheme (see :func:`jno.solve.etdrk4`)."""

    name = "etdrk4"
    continuous_flux = True

    def __init__(self, contour: int = 32):
        if int(contour) < 8:
            raise ValueError(f"jno.solve.etdrk4(contour={contour}): use at least 8 contour points (32 is the default).")
        self.contour = int(contour)

    def __repr__(self):
        return f"jno.solve.etdrk4(contour={self.contour})"


# ── term split ─────────────────────────────────────────────────────────────────────────────────────────────────


def _const(node):
    from ..trace import Constant, Literal

    n = _unwrap(node)
    if isinstance(n, (Literal, Constant)) and np.ndim(n.value) == 0:
        return complex(np.asarray(n.value)) if np.iscomplexobj(n.value) else float(np.asarray(n.value))
    if isinstance(n, (int, float)):
        return float(n)
    return None


def split_terms(node, sign=1.0):
    """``[(coefficient, term)]``: ``node`` as a sum of terms, constant factors pulled out of products and sums."""
    from ..trace import BinaryOp

    n = _unwrap(node)
    if isinstance(n, BinaryOp):
        if n.op in ("+", "-"):
            s2 = sign if n.op == "+" else -sign
            return split_terms(n.left, sign) + split_terms(n.right, s2)
        if n.op == "*":
            cl, cr = _const(n.left), _const(n.right)
            if cl is not None:
                return split_terms(n.right, sign * cl)
            if cr is not None:
                return split_terms(n.left, sign * cr)
        if n.op == "/":
            cr = _const(n.right)
            if cr is not None:
                return split_terms(n.left, sign / cr)
    return [(sign, n)]


def _rebuild(terms):
    from ..trace import BinaryOp, Literal

    out = None
    for c, t in terms:
        term = t if c == 1.0 else BinaryOp("*", Literal(c), t)
        out = term if out is None else BinaryOp("+", out, term)
    return out if out is not None else Literal(0.0)


def _classify_terms(eq, unknown, ndim):
    """Split an equation into (mass coefficient, [(coef, alpha)] linear in ``unknown``, remaining terms)."""
    from ..trace import TemporalDerivative

    def chain(node):
        return derivative_chain(node, ndim)

    mass, linear, rest = 0.0, [], []
    for c, t in split_terms(eq):
        n = _unwrap(t)
        if isinstance(n, TemporalDerivative):
            inner = chain(n.target)
            if inner is None or inner[0] is not unknown or any(inner[1]) or inner[2] is not None:
                raise NotImplementedError(
                    "jno.solve.etdrk4: the time derivative must be a plain c·u.t of the equation's unknown."
                )
            mass += c
            continue
        ch = chain(n)
        if ch is not None and ch[0] is unknown and ch[2] is None:
            linear.append((c, ch[1]))
        else:
            rest.append((c, t))
    return mass, linear, rest


# ── the march ──────────────────────────────────────────────────────────────────────────────────────────────────


def _phi_coefficients(hL, M):
    """``E, E2, Q, f1, f2, f3`` of Kassam & Trefethen for ``hL`` (complex, any shape), by the mean over ``M`` points
    of a unit circle around each value -- accumulated point by point, so memory stays that of ``hL``."""
    hL = jnp.asarray(hL, dtype=jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64)
    E, E2 = jnp.exp(hL), jnp.exp(hL / 2)
    Q = f1 = f2 = f3 = jnp.zeros_like(hL)
    for j in range(M):
        r = np.exp(2j * np.pi * (j + 0.5) / M)
        z = hL + r
        ez, ez2 = jnp.exp(z), jnp.exp(z / 2)
        Q = Q + (ez2 - 1) / z
        f1 = f1 + (-4 - z + ez * (4 - 3 * z + z * z)) / z**3
        f2 = f2 + (2 + z + ez * (-2 + z)) / z**3
        f3 = f3 + (-4 - 3 * z - z * z + ez * (4 - z)) / z**3
    return E, E2, Q / M, f1 / M, f2 / M, f3 / M


def etd_march(problem, scheme, *, x0=None, save_ts=None):
    from ..utils.solver.time_route import _infer_time_window
    from .march import initial_state

    grids = [problem._grid_of(u) for u in problem.unknowns]
    G = grids[0]
    if not fnd.all_fourier(G) or any(g != G for g in grids):
        raise NotImplementedError(
            "jno.solve.etdrk4: every unknown must be Fourier on every axis, on the same grid (a periodic box). With "
            "walls, march with jno.solve.sdirk(), bdf() or theta(0.5)."
        )
    t0, t1, dt = _infer_time_window(problem.domain)
    if dt is None:
        raise ValueError("jno.solve.etdrk4: domain.time must give n_steps >= 2.")
    nd = G.ndim
    unknowns = problem.unknowns
    diff = [k for k, u in enumerate(unknowns) if _has_temporal_of(_unwrap(problem._rows[k]["eq"]), u)]
    alg = [k for k in range(len(unknowns)) if k not in diff]
    if not diff:
        raise ValueError("jno.solve.etdrk4: no equation has a time derivative.")
    for k in diff:
        if problem._rows[k]["gauges"] or problem._rows[k]["bcs"]:
            raise NotImplementedError("jno.solve.etdrk4: a differential unknown with boundary or gauge rows.")

    def build():
        """Symbols, φ-coefficients and the compiled stepper: once per problem, step and data (cached)."""
        # linear symbols and nonlinear remainders ------------------------------------------------------------------
        sym_shape = tuple(a.n for a in G.axes[:-1]) + (G.axes[-1].n // 2 + 1,)
        Lhat, Nexpr, mass = {}, {}, {}
        for k in diff:
            m, lin, rest = _classify_terms(problem._rows[k]["eq"], unknowns[k], nd)
            if m == 0:
                raise ValueError(f"jno.solve.etdrk4: the u.t coefficient of equation {k} is zero.")
            lam = np.zeros(sym_shape, dtype=complex)
            for c, alpha in lin:
                lam = lam + c * np.broadcast_to(fnd.symbol(G, alpha), sym_shape)
            mass[k], Nexpr[k] = m, _rebuild(rest)
            Lhat[k] = jnp.asarray(-lam / m)  # u̇ = -(λ u + N)/m
        Salg, Ralg, Rspec = {}, {}, {}
        for k in alg:
            m, lin, rest = _classify_terms(problem._rows[k]["eq"], unknowns[k], nd)
            if any(_contains_unknown(t, unknowns[k]) for _c, t in rest):
                raise NotImplementedError(
                    f"jno.solve.etdrk4: the equation of algebraic unknown {k} is not linear with constant coefficients in "
                    "it; it is condensed mode by mode."
                )
            lam = np.zeros(sym_shape, dtype=complex)
            for c, alpha in lin:
                lam = lam + c * np.broadcast_to(fnd.symbol(G, alpha), sym_shape)
            Salg[k], Ralg[k] = lam, _rebuild(rest)
            # a remainder made of derivative chains of other unknowns is formed in coefficient space (no transform)
            chains = [(c, derivative_chain(t, nd)) for c, t in rest]
            if all(ch is not None and ch[2] is None for _c, ch in chains):
                Rspec[k] = [(c, ch[0].layer_id, jnp.asarray(fnd.symbol(G, ch[1]))[..., None]) for c, ch in chains]
        gauge_of = {k: [g for g, _i in problem._rows[k]["gauges"]] for k in alg}

        # helpers ------------------------------------------------------------------------------------------------------
        scope = problem._params_scope()
        lid = {k: u.layer_id for k, u in enumerate(unknowns)}

        def comps(k):
            return problem._ncomp[k]

        def gauge_value(g, co, t):
            return problem._evaluate_on(g, G, scope, t, {}, seeded=co).reshape(-1)[0]

        def solve_alg(k, co, t):
            if k in Rspec:
                rh = sum(c * S * co[j] for c, j, S in Rspec[k])
            else:
                rh = problem._equation_coeffs(problem._rows[k], Ralg[k], scope, t, {}, seeded=co)
            S = jnp.asarray(Salg[k])[..., None]
            nz = jnp.abs(S) > 0
            return jnp.where(nz, -rh / jnp.where(nz, S, 1.0), 0.0)

        # the gauge of an algebraic unknown fixes its zero-symbol (constant) mode; measured once whether the
        # mode-by-mode solution (zero mode 0) already satisfies it (a mean gauge does) and how it moves with it
        zero_mode = (0,) * nd
        gauge_plan = {}
        rng = np.random.default_rng(5)
        probe = {lid[k]: jnp.asarray(rng.standard_normal(sym_shape + (comps(k),)) * 1e-2 + 0j) for k in diff}
        for k in alg:
            probe[lid[k]] = ah = solve_alg(k, probe, 0.0)
            plans = []
            for g in gauge_of[k]:
                g0 = complex(gauge_value(g, probe, 0.0))
                g1 = complex(gauge_value(g, {**probe, lid[k]: ah.at[zero_mode].add(1.0)}, 0.0))
                kappa = g1 - g0
                if abs(kappa) == 0:
                    raise ValueError(f"jno.solve.etdrk4: the gauge row of unknown {k} does not fix its constant mode.")
                plans.append((g, kappa, abs(g0) <= 1e-12 * max(1.0, float(jnp.abs(ah).max()))))
            gauge_plan[k] = plans

        def condense(co, t):
            """Coefficients of every algebraic unknown, in declaration order, added to ``co`` ({layer_id: coeffs})."""
            co = dict(co)
            for k in alg:
                ah = solve_alg(k, co, t)
                for g, kappa, trivial in gauge_plan[k]:
                    if not trivial:
                        g0 = gauge_value(g, {**co, lid[k]: ah}, t)
                        ah = ah.at[zero_mode].add(-(g0 / kappa).real)
                co[lid[k]] = ah
            return co

        def nonlinear(vh, t):
            co = condense({lid[k]: vh[k] for k in diff}, t)
            cache = {}
            return {
                k: -problem._equation_coeffs(problem._rows[k], Nexpr[k], scope, t, cache, seeded=co) / mass[k] for k in diff
            }

        # coefficients ---------------------------------------------------------------------------------------------------
        h = float(dt)
        coef = {k: _phi_coefficients(h * Lhat[k], scheme.contour) for k in diff}
        coef = {k: tuple(c[..., None] for c in cs) for k, cs in coef.items()}  # broadcast over components

        def step(vh, t):
            Nv = nonlinear(vh, t)
            a = {k: coef[k][1] * vh[k] + h * coef[k][2] * Nv[k] for k in diff}
            Na = nonlinear(a, t + h / 2)
            b = {k: coef[k][1] * vh[k] + h * coef[k][2] * Na[k] for k in diff}
            Nb = nonlinear(b, t + h / 2)
            c = {k: coef[k][1] * a[k] + h * coef[k][2] * (2 * Nb[k] - Nv[k]) for k in diff}
            Nc = nonlinear(c, t + h)
            E, _E2, _Q, f1, f2, f3 = (None,) * 6
            out = {}
            for k in diff:
                E, _E2, _Q, f1, f2, f3 = coef[k]
                out[k] = E * vh[k] + h * (f1 * Nv[k] + 2 * f2 * (Na[k] + Nb[k]) + f3 * Nc[k])
            return out

        def snapshot(vh, t):
            co = condense({lid[k]: vh[k] for k in diff}, t)
            return [fnd.values(G, co[lid[k]]).reshape(unknowns[k].module.value.shape) for k in range(len(unknowns))]

        @jax.jit
        def advance(vh, t, nsteps):
            def body(i, carry):
                v, tt = carry
                return step(v, tt), tt + h

            return jax.lax.fori_loop(0, nsteps, body, (vh, t))

        return advance, jax.jit(snapshot)

    h = float(dt)
    key = ("etd", h, scheme.contour, problem._data_fingerprint())
    cached = problem.__dict__.setdefault("_etd_compiled", {})
    if key not in cached:
        cached[key] = build()
    advance, snapshot = cached[key]

    # start, save times, scan ------------------------------------------------------------------------------------------
    y0 = initial_state(problem, x0)
    unpacked = problem._unpack(y0)
    unpacked = unpacked if isinstance(unpacked, tuple) else (unpacked,)
    vh0 = {k: fnd.coeffs(G, jnp.asarray(unpacked[k]).reshape(G.size, problem._ncomp[k])) for k in diff}
    n_steps = int(round((t1 - t0) / h))
    ts = np.linspace(t0, t1, n_steps + 1) if save_ts is None else np.asarray(save_ts, dtype=float)
    idx = np.round((ts - t0) / h).astype(int)
    if np.any(np.abs(idx * h + t0 - ts) > 1e-9 * max(1.0, abs(t1))) or np.any(np.diff(idx) < 0) or idx[0] < 0:
        raise ValueError("jno.solve.etdrk4: save_ts must be increasing times on the domain's time grid t0 + j·dt.")
    gaps = np.diff(np.concatenate([[0], idx]))

    frames = []
    vh, t = vh0, jnp.asarray(float(t0))
    for g in gaps:
        if g:
            vh, t = advance(vh, t, int(g))
        frames.append(snapshot(vh, t))
    out = [jnp.stack([f[k] for f in frames]) for k in range(len(unknowns))]
    return out[0] if len(out) == 1 else tuple(out)
