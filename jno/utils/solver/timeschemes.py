"""Time-integration schemes for the transient path, selected with ``fem.solve(time=...)``.

Default (``time=None``): the θ-method the assembly picks — backward Euler for a first-order system,
trapezoidal for second-order. ``jno.solve.theta(θ)`` overrides θ (``1`` backward Euler, ``1/2``
Crank–Nicolson, ``0`` forward Euler). ``jno.solve.exponential(...)`` advances a linear parabolic block
with **time-independent** ``M``, ``A`` by the **matrix exponential** — its homogeneous decay exact in time
and unconditionally stable, so it takes large stiff steps that an implicit θ-step cannot (a time-varying
source is integrated by ETD2). For a symmetric operator it reuses the Lanczos
:func:`jno.solve.applyfun`; for a **non-symmetric** one (``symmetric=False``, advection–diffusion) it uses
an Arnoldi + differentiable **Padé** exponential (:func:`jno.utils.solver.matfun.expmv`), all matfree.
Step **size** is a separate axis from which step to take, so it composes rather than being its own scheme:
``.adaptive(...)`` on any scheme chooses the step size from a **step-doubling** local-error estimate of
*that* scheme's step, working for every transient case (real/complex, scalar/vector, plain/periodic) and
staying reverse-mode differentiable (a fixed-budget ``lax.scan`` with the controller ``stop_gradient``-ed —
the state differentiates at the realized step schedule)::

    fem.solve(time=jno.solve.theta(0.5))                       # 2nd-order, fixed grid
    fem.solve(time=jno.solve.theta(0.5).adaptive(rtol=1e-5))   # 2nd-order, error-controlled
    fem.solve(time=jno.solve.adaptive(rtol=1e-5))              # the block's own θ-step, error-controlled

The controller's step-size exponent ``1/(p+1)`` follows the base scheme's ``step_order``, so attaching it
to a second-order step is far cheaper per digit than the first-order default — see :func:`jno.solve.adaptive`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class _TimeScheme:
    """Base for the ``fem.solve(time=…)`` schemes.

    A time scheme has two separable jobs: **which step** to take (a θ-step, a matrix exponential, later
    an SDIRK / Rosenbrock stage) and **how big** the steps are (the domain's fixed grid, or chosen from an
    error estimate). ``integrate`` marches on the fixed grid; ``.adaptive(...)`` wraps *this same base
    step* in the step-doubling controller, so the two axes compose instead of each scheme carrying its own
    copy of the other::

        fem.solve(time=jno.solve.theta(0.5))                       # 2nd-order, fixed grid
        fem.solve(time=jno.solve.theta(0.5).adaptive(rtol=1e-5))   # 2nd-order, error-controlled

    A subclass supplies ``step_order`` — the classical order ``p`` of its base step, which the controller
    needs for its ``1/(p+1)`` exponent — and ``stepper``, a single ``(u, t, dt) -> u(t+dt)``."""

    step_order = 1

    def stepper(self, block, args, *, linear_solve=None, nonlinear_solve=None):
        """One implicit step of THIS scheme, as ``(u, t, dt) -> u(t+dt)``. Schemes that cannot expose a
        single arbitrary-``dt`` step (and so cannot be adaptively sized) leave this raising."""
        raise NotImplementedError(f"{type(self).__name__} does not expose a single step, so it cannot be adaptively sized.")

    def adaptive(self, *, rtol: float = 1e-4, atol: float = 1e-6, max_steps: int = 1000, dt0: float | None = None):
        """Size THIS scheme's step from a step-doubling local-error estimate. See
        :func:`jno.solve.adaptive` for the tolerances, the step budget, and the ``dt0`` rule."""
        return _AdaptiveScheme(self, rtol, atol, max_steps, dt0)


class _ThetaScheme(_TimeScheme):
    """θ-method scheme (see :func:`jno.solve.theta`) — reuses the default stepper with an explicit θ."""

    def __init__(self, theta: float):
        self.theta = float(theta)

    @property
    def step_order(self):
        return _theta_order(self.theta)

    def stepper(self, block, args, *, linear_solve=None, nonlinear_solve=None):
        def step_fn(u, t, dt):
            return block.step(
                u, t, dt, args=args, theta=self.theta, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve
            )

        return step_fn

    def integrate(self, block, args, save_ts, *, linear_solve, nonlinear_solve):
        from .backend_blocks import _default_transient_integrate

        return _default_transient_integrate(
            block, args, save_ts, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve, theta=self.theta
        )

    def __repr__(self):
        return f"jno.solve.theta({self.theta})"


class _ExponentialScheme(_TimeScheme):
    """Matrix-exponential scheme (see :func:`jno.solve.exponential`) — linear, time-independent ``M``/``A``
    (a time-varying source is handled by ETD2).

    NOTE ``order`` here is the **Krylov subspace size**, not the method's order in ``dt`` — do not feed it
    to the adaptive controller."""

    def __init__(self, order: int, mass: str, symmetric: bool):
        self.order = int(order)
        self.mass = mass
        self.symmetric = bool(symmetric)

    def adaptive(self, **kwargs):
        raise NotImplementedError(
            "jno.solve.exponential(...).adaptive(...) is not available: this scheme is already exact in time "
            "for the homogeneous decay, so a step-doubling estimate would measure only the ETD2 source term "
            "and would size steps by that alone. Use jno.solve.theta(...).adaptive(...)."
        )

    def integrate(self, block, args, save_ts, *, linear_solve=None, nonlinear_solve=None):
        return _exponential_integrate(block, args, save_ts, order=self.order, mass=self.mass, symmetric=self.symmetric)

    def __repr__(self):
        return f"jno.solve.exponential(order={self.order}, mass={self.mass!r}, symmetric={self.symmetric})"


class _AdaptiveScheme(_TimeScheme):
    """Step-size **policy** wrapping a base scheme — step-doubling (Richardson) error control on that
    scheme's implicit step, so it inherits the block's DAE handling and works for a linear or nonlinear
    block, scalar or vector, plain or periodic-reduced (the step runs in the reduced space; the caller
    prolongs). The complex-transient path feeds the same marcher its 2n-block step.

    Built by :meth:`_TimeScheme.adaptive` (``jno.solve.theta(0.5).adaptive(...)``). ``base=None`` — what
    the bare :func:`jno.solve.adaptive` produces — means "whatever θ-step the assembly picked for this
    block", which is backward Euler for a parabolic block and trapezoidal for a second-order one."""

    def __init__(self, base, rtol: float, atol: float, max_steps: int, dt0: float | None = None):
        self.base = base
        self.rtol, self.atol, self.max_steps = float(rtol), float(atol), int(max_steps)
        self.dt0 = None if dt0 is None else float(dt0)

    def base_for(self, block):
        """The base scheme this will actually step with — ``self.base``, or the block's own θ-step."""
        if self.base is not None:
            return self.base
        theta = float(block.metadata.get("theta", 1.0)) if getattr(block, "metadata", None) else 1.0
        return _ThetaScheme(theta)

    def adaptive(self, **kwargs):
        raise NotImplementedError("this scheme is already adaptively sized; .adaptive() does not nest.")

    def integrate(self, block, args, save_ts, *, linear_solve=None, nonlinear_solve=None):
        s0f = getattr(block, "state0_fn", None)
        u0 = jnp.asarray(s0f(args) if s0f is not None else block.state0).reshape(-1)
        base = self.base_for(block)
        step_fn = base.stepper(block, args, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve)

        # NOTE: deliberately NOT ``block.dt``. The output grid's dt has no relation to the correct step
        # size, and with no rejection an over-large first step is committed for good — ``dt0=None`` lets
        # the marcher start at its floor and grow into the right scale. See ``adaptive_march``.
        return adaptive_march(
            step_fn,
            u0,
            float(block.t0),
            float(block.t1),
            save_ts,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            dt0=self.dt0,
            order=base.step_order,
        )

    def __repr__(self):
        base = "" if self.base is None else f"{self.base!r}."
        return f"{base}adaptive(rtol={self.rtol}, atol={self.atol}, max_steps={self.max_steps}, dt0={self.dt0})"


def _theta_order(theta: float) -> int:
    """Classical order of the θ-step: the θ-method is O(dt²) **only** at θ=1/2 (Crank–Nicolson /
    trapezoidal) and O(dt) everywhere else. The step-size controller needs this to pick its exponent
    ``1/(p+1)``; getting it wrong does not break the march, it just sizes every step badly."""
    return 2 if abs(float(theta) - 0.5) < 1e-12 else 1


def _safe_interp(x, xp, fp):
    """Piecewise-linear interpolation that stays gradient-safe when ``xp`` has repeated (flat) values.
    A rejected or settled adaptive step records a **duplicate** time, and ``jnp.interp`` divides by the
    zero gap in its VJP → ``NaN``; here a flat gap contributes zero slope, so ``∂/∂fp`` stays finite."""
    n = xp.shape[0]
    i = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, n - 2)
    x0, x1, f0, f1 = xp[i], xp[i + 1], fp[i], fp[i + 1]
    gap = x1 - x0
    w = jnp.where(gap > 0, (x - x0) / jnp.where(gap > 0, gap, 1.0), 0.0)  # zero slope on a flat gap (no 0/0)
    return f0 + w * (f1 - f0)


def adaptive_march(
    step_fn,
    u0,
    t0,
    t1,
    save_ts,
    *,
    rtol,
    atol,
    max_steps,
    dt0=None,
    order=1,
    safety=0.9,
    min_factor=0.2,
    max_factor=5.0,
):
    """Adaptive step-size march via **step doubling** (Richardson error control), reverse-mode
    differentiable. ``step_fn(u, t, dt) -> u(t+dt)`` is one implicit step (the block's DAE-correct
    θ-step, or the complex 2n-block step). Each attempt compares a single full step with two half-steps;
    the normalized RMS difference sizes the next dt (exponent ``1/(order+1)`` — ½ for a first-order base
    such as backward Euler, ⅓ for a second-order one such as trapezoidal / Crank–Nicolson). The march is a
    **fixed-length** ``lax.scan`` of ``max_steps`` — a static trip count, so the whole thing stays
    reverse-differentiable — where the settled / over-budget tail simply consumes an iteration without
    advancing ``t``. The trajectory is sampled at ``save_ts`` by interpolation; if the budget is exhausted
    before ``t1`` the result is **NaN-poisoned** (raise ``max_steps``) rather than silently under-resolving
    the tail — jNO never fails silently.

    **Every attempt is accepted** — see the ``body`` comment for why rejection is not available here. The
    consequence is that ``dt0`` must not overshoot: an over-large step is committed, and only the *next* one
    shrinks (and by at most ``1/min_factor`` per attempt). ``dt0=None`` (the default) therefore starts at
    the floor ``1e-4·span`` and lets the controller **grow** into the right scale at up to ``max_factor``
    per attempt — an under-sized step costs work, never accuracy, so approaching from below cannot commit
    an out-of-tolerance step. Passing the caller's output-grid ``dt`` instead measured **4.2x** worse on a
    2-D heat benchmark, its first two attempts landing 741x and 34x over tolerance and staying in the
    answer."""
    u0 = jnp.asarray(u0).reshape(-1)
    rt = jnp.real(u0).dtype  # time / dt live in the state's real dtype (a complex system marches its real 2n block)
    t0, t1 = jnp.asarray(t0, rt), jnp.asarray(t1, rt)
    span = jnp.abs(t1 - t0)
    # A non-degenerate floor: a *tiny* dt makes ``M + dt·A`` near-singular on the zeroed Dirichlet rows
    # (its adjoint then blows up to NaN), so never step below this — the fixed-length scan already bounds
    # the count, so the floor needs only to keep every step well-conditioned, not to bound work.
    dt_min = 1e-4 * span
    dt_max = span
    # Approach the step size from BELOW by default (see the docstring): growth is safe, overshoot is not.
    dt_start = dt_min if dt0 is None else jnp.clip(jnp.asarray(dt0, rt), dt_min, dt_max)

    def _err(a, b):  # normalized RMS of the two estimates' difference (mixed absolute / relative tolerance)
        scale = atol + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b))
        return jnp.sqrt(jnp.mean(jnp.abs((a - b) / scale) ** 2))

    def _take(operand):
        u, _t, _dt = operand
        # ``u_full`` is used ONLY to size the next step, so freeze its gradient entirely — otherwise its
        # matrix-free solve adjoint would run with a zero cotangent and hit a 0/0 (‖r‖/‖b‖ with b=0) → NaN.
        u_full = jax.lax.stop_gradient(step_fn(u, _t, _dt))
        u_half = step_fn(step_fn(u, _t, 0.5 * _dt), _t + 0.5 * _dt, 0.5 * _dt)  # the accepted (more accurate) state
        return u_half, jax.lax.stop_gradient(_err(u_full, u_half))

    def body(carry, _):
        t, u, dt = carry  # t, dt are the step SCHEDULE (control); the gradient flows through the state u only
        remaining = t1 - t
        done = remaining <= dt_min
        dt_ctrl = jnp.clip(dt, dt_min, dt_max)
        dt_step = jnp.clip(jnp.minimum(dt_ctrl, remaining), dt_min, dt_max)  # land on t1; never a degenerate dt
        # Differentiate the numerical state at the REALIZED (t, dt) schedule; the controller (the error
        # estimate and the next dt) is a discrete control decision whose derivative is both fragile (√err
        # blows up exactly when a step is well-resolved) and not what an inverse problem wants, so
        # ``stop_gradient`` it — the standard way to differentiate an adaptive solver (Diffrax does the
        # same): the gradient goes through u and the operator params at the step sequence the forward pass
        # chose. On a **settled** step ``lax.cond`` runs the *skip* branch (no solve), so no zero-cotangent
        # solve adjoint runs there.
        #
        # **No rejection**, and the reason is AD, not stability: rejecting would mean discarding this
        # attempt's ``u_half`` (``where(accept, u_half, u)``), whose cotangent is then exactly zero, so the
        # matrix-free solve adjoint runs on ``b = 0`` and its relative-residual test divides 0/0. Measured:
        # adding rejection turns a finite gradient (AD -21.92 vs FD -21.99) into ``NaN``. Accepting every
        # attempt keeps ``u_half`` a genuinely-used state. NOTE the older rationale here — "a too-large step
        # is only inaccurate, never unstable (backward Euler is L-stable)" — argued the wrong thing:
        # L-stability says the step will not blow up, not that an out-of-tolerance step is acceptable to
        # keep. Since it IS kept, the burden falls on ``dt0`` never overshooting (see the docstring).
        _t, _dt = jax.lax.stop_gradient(t), jax.lax.stop_gradient(dt_step)
        u_new, err = jax.lax.cond(done, lambda o: (o[0], jnp.asarray(0.0, rt)), _take, (u, _t, _dt))
        t_new = jnp.where(done, t, t + _dt)
        # Exponent 1/(p+1) for a base method of order p — NOT a hardwired ½. A second-order base (θ=1/2,
        # including the trapezoidal step the assembly picks for second-order systems) needs ⅓, or every
        # step is mis-sized.
        fac = jnp.clip(safety * jnp.maximum(err, 1e-12) ** (-1.0 / (int(order) + 1)), min_factor, max_factor)
        dt_next = jnp.clip(dt_ctrl * fac, dt_min, dt_max)
        return (t_new, u_new, dt_next), (t_new, u_new)

    (t_end, _u, _dt), (ts, us) = jax.lax.scan(body, (t0, u0, dt_start), None, length=int(max_steps))
    ts = jax.lax.stop_gradient(jnp.concatenate([t0[None], ts]))  # sample times are a fixed schedule (control)
    us = jnp.concatenate([u0[None, :], us], axis=0)
    save = jnp.asarray(save_ts, rt)
    out = jax.vmap(lambda col: _safe_interp(save, ts, col), in_axes=1, out_axes=1)(us)
    reached = t_end >= t1 - 2.0 * dt_min  # the last real step lands within dt_min of t1 (see ``done``)
    # Fail loud if max_steps was too small -- and fail loud in the ADJOINT too. ``where(reached, out, nan)``
    # would poison only the value: the VJP of ``where`` w.r.t. its taken branch is ``where(c, g, 0)``, so an
    # unreached march returns a NaN trajectory whose gradient is exactly **zero**. An inverse problem reads
    # the gradient, not the value, and a plausible zero reads as "converged" -- the budget being too small
    # would look like a converged optimisation. Multiplying instead carries the NaN into the cotangent
    # (``d(out·c)/d(out) = c``), so the failure survives differentiation. Exact no-op when reached: the
    # factor is 1.0 in both the primal and the adjoint.
    #
    # KNOWN GAP: this fixes the poison at *this* site only. Something further down the transient adjoint
    # still scrubs a NaN cotangent -- an unconditional ``out * nan`` here yields a NaN trajectory with a
    # *finite* parameter gradient, while the same construction over a plain ``lax.scan`` propagates NaN
    # correctly. Until that is located, a starved budget is loud in the value and unreliable in the
    # gradient: CHECK THE VALUE. See plans/adaptive-api-and-differentiability.md, stage 1.
    poison = jnp.where(reached, jnp.asarray(1.0, rt), jnp.asarray(jnp.nan, rt))
    return out * poison


def _exponential_integrate(block, args, save_ts, *, order, mass, symmetric):
    """Advance ``M u̇ + A u = f(t)`` per step via ``exp(-dt·M⁻¹A)``: exact for the homogeneous decay, with a
    ``φ₁`` weight for a constant source and a ``φ₂`` ramp weight (ETD2) for a time-varying one.

    * ``lumped`` — symmetric ``L̃ = D^{-1/2} A D^{-1/2}`` (``D`` = row-sum mass); integrate ``w = D^{1/2} u``
      with the Euclidean Lanczos of :func:`applyfun`. Matrix-free.
    * ``consistent`` — ``L = M⁻¹A`` applied by a **matrix-free M-inner-product Lanczos** (:func:`m_inner_funm`,
      an M-solve per matvec via CG). No lumping error, no factorization.

    Both are **matrix-free and reverse-mode differentiable**; the Dirichlet boundary (zero-mass DOFs) is
    held at 0 by *masking* (a multiply — trace-safe), never a host-side extraction."""
    from .backend_blocks import _block_time_grid
    from .mass import lumped_diagonal, m_inner_funm
    from .matfun import applyfun, expmv
    from .solver_api import LinearOperator

    if not block.is_linear():
        raise NotImplementedError(
            "jno.solve.exponential integrates a LINEAR block only; use a θ-scheme for a nonlinear one."
        )
    if block.operator_fn is not None or block.mass_fn is not None:
        raise NotImplementedError(
            "jno.solve.exponential needs time-INDEPENDENT M and A (an autonomous system). For time-varying "
            "coefficients use jno.solve.theta(...)."
        )
    if mass not in ("lumped", "consistent"):
        raise ValueError(f"jno.solve.exponential: mass={mass!r} — use 'lumped' or 'consistent'.")

    Mop, Aop = block.M, block.A
    s0 = jnp.asarray(block.state0).reshape(-1)
    dtype = s0.dtype
    n = s0.shape[0]
    dt = float(block.dt)
    grid = jnp.asarray(_block_time_grid(block), dtype)

    # Forcing = a CONSTANT part (affine_bias) + an optional time-varying source f(t). A time-INDEPENDENT
    # problem uses the fast φ₁ path with a single precomputed vector; a time-VARYING source is integrated by
    # ETD2 (the exponential trapezoidal rule): the source is sampled at both ends of each step and its ramp
    # enters through a φ₂ weight -- EXACT for a source affine in time, second-order for a general one
    # (Hochbruck & Ostermann, "Exponential integrators", Acta Numerica 19 (2010) 209-286, §2.3). The
    # homogeneous decay stays exact-in-time either way.
    c_aff = jnp.zeros(n, dtype)  # the constant (affine_bias) forcing
    if block.affine_bias is not None:
        c_aff = c_aff + jnp.asarray(block.affine_bias, dtype).reshape(-1)
    _time_varying = block.forcing_vector_fn is not None
    has_forcing = bool(block.affine_bias is not None or _time_varying)

    def _f_of(t):  # total forcing vector at time t: constant affine_bias + the time-varying source
        f = c_aff
        if _time_varying:
            f = f + jnp.asarray(block.forcing_vector_fn(t, args or {}), dtype).reshape(-1)
        return f

    def _exp(lam):
        return jnp.exp(-dt * lam)

    def _phi1(lam):  # φ₁(z) = (eᶻ − 1)/z at z = −dt·λ (Taylor near 0 for stability)
        z = -dt * lam
        return jnp.where(jnp.abs(z) < 1e-7, 1.0 + 0.5 * z, jnp.expm1(z) / z)

    def _phi2(lam):  # φ₂(z) = (eᶻ − 1 − z)/z² at z = −dt·λ (Taylor near 0) — the ETD2 ramp weight
        z = -dt * lam
        return jnp.where(jnp.abs(z) < 1e-5, 0.5 + z / 6.0, (jnp.expm1(z) - z) / (z * z))

    d = lumped_diagonal(Mop)  # Dirichlet DOFs carry NO mass (d=0) — algebraic (u=0), not ODEs
    mask = (d > 1e-12 * jnp.max(d)).astype(dtype)  # 1 interior / 0 boundary — a *multiply* (trace-safe)

    def _consistent_m_solve():
        """A masked, Jacobi-preconditioned CG solve for ``M⁻¹·(mask·rhs)`` — used by the consistent-mass
        paths (boundary → 0, since the masked rhs vanishes there). Reverse-mode differentiable."""
        from jax.scipy.sparse.linalg import cg as _cg

        from .linear import matrix_diagonal

        Mreg = lambda x: Mop @ x + (1.0 - mask) * x  # M + (1-mask)·I — SPD (boundary block is identity)
        jac = 1.0 / (matrix_diagonal(Mop) + (1.0 - mask))  # Jacobi preconditioner for the CG M-solve
        return lambda rhs: _cg(Mreg, mask * rhs, tol=1e-10, maxiter=300, M=lambda z: jac * z)[0]

    if not symmetric:
        # NON-symmetric A (advection–diffusion): L = M⁻¹A is not self-adjoint, so neither the symmetric
        # similarity nor the M-inner Lanczos applies. Advance with exp(-dt·L) by **Arnoldi + a differentiable
        # Padé** exponential (:func:`expmv`, GPU + reverse-mode diff). Forcing enters *exactly* through an
        # augmented generator: a CONSTANT source rides one row (G = [[-L, g], [0, 0]] → φ₁), and a
        # TIME-VARYING one adds a ramp row (G = [[-L, g₀, g₁], [0,0,0], [0,1,0]] → φ₁ + φ₂, ETD2), so one
        # exponential covers the homogeneous decay and the forcing, and ‖[x; 1; …]‖ ≥ 1 never vanishes.
        if mass == "lumped":
            d_inv = jnp.where(mask > 0, 1.0 / jnp.where(mask > 0, d, 1.0), 0.0)  # 0 on the boundary
            m_inv = lambda r: d_inv * r
        else:
            m_solve = _consistent_m_solve()
            m_inv = m_solve
        L_mv = lambda x: m_inv(Aop @ x)  # M⁻¹A x (interior; boundary held at 0 by m_inv)
        w0, to_field = mask * s0, lambda u: u
        if not _time_varying:
            g = m_inv(c_aff)  # constant forcing (fast φ₁ path)

        def step(w, t_target):
            if _time_varying:
                f0, f1 = _f_of(t_target - dt), _f_of(t_target)  # source at both step ends
                g0, g1 = m_inv(f0), m_inv(f1 - f0)  # constant part (φ₁) + ramp increment (φ₂)
                y0 = jnp.concatenate([w, jnp.ones((1,), dtype), jnp.zeros((1,), dtype)])  # [x; a=1; b=0]

                def gen(y):  # dt·G·[x; a; b]:  ẋ = -L x + a·g₀ + b·g₁ ;  ȧ = 0 ;  ḃ = a
                    x, a, b = y[:n], y[n], y[n + 1]
                    return jnp.concatenate([dt * (-L_mv(x) + a * g0 + b * g1), jnp.zeros((1,), dtype), y[n : n + 1]])

                wn = expmv(LinearOperator.from_matvec(gen, shape=(n + 2, n + 2)), y0, order=order)[:n]
            else:
                y0 = jnp.concatenate([w, jnp.ones((1,), dtype)])  # augment with the constant "1" row

                def gen(y):  # dt·G·[x; a] = dt·[-L x + a·g ; 0]
                    x, a = y[:n], y[n]
                    return dt * jnp.concatenate([-L_mv(x) + a * g, jnp.zeros((1,), dtype)])

                wn = expmv(LinearOperator.from_matvec(gen, shape=(n + 1, n + 1)), y0, order=order)[:n]
            return wn, wn
    else:
        if mass == "lumped":
            # symmetric L̃ = D^{-1/2} A D^{-1/2}; integrate w = D^{1/2} u with the Euclidean Lanczos of applyfun
            s = jnp.sqrt(d)
            s_inv = jnp.where(mask > 0, 1.0 / jnp.where(mask > 0, s, 1.0), 0.0)  # 0 on the boundary
            Ctil = LinearOperator.from_matvec(lambda w: s_inv * (Aop @ (s_inv * w)), shape=(n, n))
            e0 = mask / jnp.linalg.norm(mask)

            def apply_f(vec, fun):  # f(L̃)·vec, on the *normalised* vector (Lanczos divides by ‖·‖ → NaN at 0)
                nrm = jnp.linalg.norm(vec)
                unit = jnp.where(nrm > 1e-300, vec / jnp.where(nrm > 1e-300, nrm, 1.0), e0)
                return nrm * applyfun(Ctil, unit, fun=fun, order=order)

            w0, solve_c = s * s0, lambda x: s_inv * x  # forcing in the scaled system: D^{-1/2}·rhs
            to_field = lambda w: s_inv * w  # u = D^{-1/2} w (boundary → 0)
        else:  # consistent — matrix-free M-inner-product Lanczos on L = M⁻¹A: scalable AND differentiable
            m_solve = _consistent_m_solve()
            L_mv = lambda x: m_solve(Aop @ x)  # M⁻¹A x (interior)
            m_inner = lambda a, b: a @ (Mop @ b)  # ⟨a,b⟩_M — M already zeros the boundary
            e0 = mask / jnp.sqrt(m_inner(mask, mask))  # a fixed M-unit interior vector
            apply_f = lambda vec, fun: m_inner_funm(L_mv, m_inner, e0, vec, fun, order)
            w0, solve_c = mask * s0, m_solve  # integrate u directly (homogeneous boundary); forcing M⁻¹·rhs
            to_field = lambda u: u
        if not _time_varying:
            g = solve_c(c_aff)  # constant forcing (fast φ₁ path)

        def step(w, t_target):
            wn = apply_f(w, _exp)
            if has_forcing:
                if _time_varying:  # ETD2: constant part rides φ₁, the step's source ramp rides φ₂
                    f0, f1 = _f_of(t_target - dt), _f_of(t_target)
                    wn = wn + dt * apply_f(solve_c(f0), _phi1) + dt * apply_f(solve_c(f1 - f0), _phi2)
                else:
                    wn = wn + dt * apply_f(g, _phi1)
            return wn, wn

    _, ws = jax.lax.scan(step, w0, grid[1:])
    traj_u = jax.vmap(to_field)(jnp.concatenate([w0[None, :], ws], axis=0))  # each row → the full field u
    save_ts = jnp.asarray(save_ts, dtype)
    return jax.vmap(lambda col: jnp.interp(save_ts, grid, col), in_axes=1, out_axes=1)(traj_u)
