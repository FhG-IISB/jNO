"""The transient spectral problem: the method of lines on jNO's own time schemes, or ETDRK4.

``M u̇ + R(u, t) = 0`` with ``R`` the spatial residual (``u.t`` dropped), ``M`` diagonal: the ``u.t`` coefficient on
the equation rows of a differential unknown, zero on boundary and gauge rows and on the rows of an algebraic
unknown (one whose equation has no ``u.t``, e.g. a streamfunction). Any ``jno.solve`` time scheme marches it
through :class:`~jno.utils.solver.backend_blocks.SemidiscreteTimeBlock` (backward Euler by default);
``jno.solve.etdrk4()`` takes the exponential route of :mod:`jno._spectral.etd`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .._strong_form import _has_temporal_of, _set_temporal, _unwrap

__all__ = ["march", "initial_state", "mass_diagonal"]


def _module_values(problem, dofs):
    return {**problem._params_scope(), **problem._inject(dofs)}


def initial_state(problem, x0=None):
    """The start state: ``x0`` (nodal values per unknown, as ``solve`` returns them, or the DOF vector), else each
    initial condition ``U(x0, y0) - g`` solved node by node (affine in ``U``); unknowns without one start at 0."""
    n = problem._Ntot
    if x0 is not None:
        return jnp.asarray(x0) if np.ndim(x0) == 1 and np.size(x0) == n else problem._pack(x0)
    zero, one = jnp.zeros(n), jnp.ones(n)
    y = jnp.zeros(n)
    by_field = {}
    for c in problem._ic:
        by_field.setdefault(problem._field_of(c), []).append(c)
    for k, cs in by_field.items():
        if len(cs) > 1:
            raise ValueError(f"jno.spectral([...]): unknown {k} has {len(cs)} initial conditions; give one.")
        c = cs[0]
        own = problem._grid_of(problem.unknowns[k])
        if problem._grid_of_expr(c, own) != own:
            raise NotImplementedError(
                "jno.spectral([...]): an initial condition must be written on the unknown's own nodes."
            )
        r0 = problem._evaluate_on(c, own, _module_values(problem, zero), None, {})
        r1 = problem._evaluate_on(c, own, _module_values(problem, one), None, {})
        a = r1 - r0
        if not bool(jnp.all(jnp.abs(a) > 0)):
            raise ValueError(
                f"jno.spectral([...]): the initial condition of unknown {k} does not determine it at every node."
            )
        u0 = -r0 / a  # (size, c)
        o, sz, cc = problem._offsets[k], problem._sizes[k], problem._ncomp[k]
        y = y.at[o : o + sz * cc].set(u0.T.reshape(-1))
    return y


def mass_diagonal(problem):
    """``(Ntot,)``: the ``u.t`` coefficient per row (constant in time and state), zero on algebraic rows."""
    n = problem._Ntot
    params = _module_values(problem, jnp.zeros(n))
    parts = []
    for k, (u, r) in enumerate(zip(problem.unknowns, problem._rows)):
        eq = _unwrap(r["eq"])
        size, c = problem._sizes[k], problem._ncomp[k]
        if not _has_temporal_of(eq, u):
            parts.append(jnp.zeros(size * c))
            continue
        cache = {}
        f1 = problem._equation_values(r, _set_temporal(eq, 1.0), params, None, cache)
        f0 = problem._equation_values(r, _set_temporal(eq, 0.0), params, None, cache)
        m = jnp.broadcast_to(f1 - f0, (r["grid"].size, c))
        if not r["replace"]:
            raise NotImplementedError("jno.spectral([...]): a transient equation on another node set than its unknown's.")
        for _kind, _c, idx in r["bcs"]:
            m = m.at[jnp.asarray(idx)].set(0.0)
        for _g, i in r["gauges"]:
            m = m.at[i].set(0.0)
        parts.append(m.T.reshape(-1))
    return jnp.concatenate(parts)


def march(problem, *, time=None, x0=None, linear=None, precond=None, save_ts=None, nonlinear=None, span=None, raw=False):
    """The march; ``span=(t0, t1)`` marches that window at the domain's step instead (a coupling window), ``raw``
    returns the DOF-vector trajectory ``(n_saves, Ntot)`` instead of the per-unknown arrays."""
    from .etd import ETDRK4Scheme, etd_march

    if problem._time_order != 1:
        raise NotImplementedError("jno.spectral([...]): u.tt (second order in time) is not supported yet.")
    if isinstance(time, ETDRK4Scheme):
        if span is not None or raw:
            raise NotImplementedError(
                "jno.solve.etdrk4 marches a problem that is Fourier on every axis; such a box has no face to couple "
                "through, so a coupling window cannot use it. March the coupled problem with jno.solve.theta(0.5), "
                "bdf2() or sdirk()."
            )
        return etd_march(problem, time, x0=x0, save_ts=save_ts)
    import jax.experimental.sparse as jsparse

    from ..fdm import _integrate_transient
    from ..utils.solver.backend_blocks import SemidiscreteTimeBlock, _block_time_grid
    from ..utils.solver.time_route import _infer_time_window

    t0, t1, dt = _infer_time_window(problem.domain)
    if dt is None:
        raise ValueError("jno.spectral([...]): domain.time must give n_steps >= 2 for a transient march.")
    if span is not None:
        t0, t1 = float(span[0]), float(span[1])
    y0 = initial_state(problem, x0)
    n = problem._Ntot
    md = mass_diagonal(problem)
    diag = jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1)
    M = jsparse.BCOO((md, diag), shape=(n, n))
    R = jax.jit(problem._residual_fn(spatial=True))
    block = SemidiscreteTimeBlock(
        mass=lambda t, args: M,
        residual=lambda y, t, args: R(y, t),
        state0=y0,
        t0=float(t0),
        t1=float(t1),
        dt=float(dt),
        metadata={},
    )
    ts = _block_time_grid(block) if save_ts is None else jnp.asarray(save_ts)
    step_solve = None
    if (
        nonlinear is None and linear is None and precond is not None
    ):  # opt-in: measured slower than the default at 16x17..64x65
        step_solve = _preconditioned_newton(problem, R, md, y0, float(t0), _stage_scale(time, float(dt)), precond)
    traj = _integrate_transient(block, ts, time, None, step_solve if step_solve is not None else nonlinear)
    return traj if raw else problem._unpack(traj)


def _stage_scale(time, dt):
    """``s`` with the implicit stage operator ``∝ M + s ∂R/∂u`` of a jno time scheme (backward Euler by default):
    ``θ dt`` (theta), ``2 dt / 3`` (BDF2), ``γ dt`` (SDIRK). Any other scheme gets ``dt``: the preconditioner is
    then approximate and GMRES makes up the rest."""
    name = type(time).__name__
    if time is None:
        return dt
    if name == "_ThetaScheme":
        return max(float(time.theta), 1e-12) * dt
    if name == "_BDF2Scheme":
        return 2.0 * dt / 3.0
    if name == "_SDIRKScheme":
        return float(time.gamma) * dt
    return dt


def _preconditioned_newton(problem, R, md, y0, t0, scale, precond):
    """The per-step solve of a march: Newton, its inner GMRES preconditioned by the Fourier-mode block inverse
    of ``M + s ∂R/∂u`` frozen at the start state (built once, before the march; exact for a linear problem with
    coefficients constant along the Fourier axes, so a step costs a GMRES iteration or two). ``None`` when the
    blocks do not apply (unknowns without shared Fourier axes, a block too large)."""
    from .. import solve as _solve
    from ..fdm import _gmres_incremental
    from .precond import SpectralPrecond

    spec = precond if isinstance(precond, SpectralPrecond) else SpectralPrecond()
    J0 = lambda v: jax.jvp(lambda y: R(y, t0), (y0,), (v,))[1]  # noqa: E731
    try:
        P = spec.build(problem, lambda v: md * v + scale * J0(v))
    except NotImplementedError:
        if precond is not None:
            raise
        return None
    gm = _solve.gmres(tol=1e-11, maxiter=60, restart=30)

    def newton(G, u, jacobian=None, **_):
        def body(state):
            w, _r, k = state
            r = G(w)
            dw = _gmres_incremental(lambda v: jax.jvp(G, (w,), (v,))[1], -r, P, gm)[0]
            return w + dw, jnp.linalg.norm(r), k + 1

        def cond(state):
            w, rn, k = state
            return (k < 20) & ((k == 0) | (rn > 1e-10 * (1.0 + jnp.linalg.norm(w))))

        w, _rn, _k = jax.lax.while_loop(cond, body, (u, jnp.asarray(jnp.inf, u.dtype), 0))
        return w

    return newton
