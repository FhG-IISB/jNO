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
import numpy as np


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
        def step_fn(u, t, dt, guess=None):
            return block.step(
                u, t, dt, args=args, theta=self.theta, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve,
                guess=guess,
            )  # fmt: skip

        return step_fn

    def integrate(self, block, args, save_ts, *, linear_solve, nonlinear_solve):
        from .backend_blocks import _default_transient_integrate

        return _default_transient_integrate(
            block, args, save_ts, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve, theta=self.theta
        )

    def step_scales(self, block):
        """The coefficient of ``A`` in this scheme's step operator ``M + θ·Δt·A``, so a composed linear
        solve pre-builds THAT matrix. Without it only the block's default (θ = 1) was built, and a
        Crank–Nicolson step was solved with the backward-Euler matrix (measured: 0.34 off at T)."""
        return (self.theta * float(block.dt),) if block.dt is not None else ()

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

    def preflight(self, block):
        """Run the structural refusals once, eagerly, on the built block (at ``fem.solve`` time).

        Inside ``jno.core`` the march is traced, the values the guards read are tracers, and they would skip:
        measured, a lumped Hermite march then returned NaN instead of the refusal it gets when solved eagerly."""
        stored = _stored_runtime_args(block)
        if stored is None:  # a trainable with no stored value to evaluate at: the in-trace checks are all there is
            return
        _exponential_integrate(
            block, stored, None, order=self.order, mass=self.mass, symmetric=self.symmetric, check_only=True
        )

    def __repr__(self):
        return f"jno.solve.exponential(order={self.order}, mass={self.mass!r}, symmetric={self.symmetric})"


class _BDF2Scheme(_TimeScheme):
    """Second-order backward differentiation formula; see :func:`jno.solve.bdf2`.

    ``(3u^{n+1} - 4u^n + u^{n-1}) / (2 dt) M + R(u^{n+1}) = 0``.

    Implemented by REDUCTION rather than by a second stepper: scaled by ``2dt/3`` the BDF2 step is
    *exactly* a backward-Euler step with step ``dt_eff = 2dt/3`` taken from the shifted state
    ``u* = (4u^n - u^{n-1})/3``, landing at the same ``t + dt``::

        (M + (2dt/3) A) u^{n+1} = M (4u^n - u^{n-1})/3 + (2dt/3)(c + f)

    So it reuses :meth:`SemidiscreteTimeBlock.step` unchanged, and inherits everything that lives
    there: the Dirichlet/DAE row handling, the solver-slot injection, the ``metadata["krylov"]``
    complex path and the verified BiCGStab->GMRES rescue. Nothing is duplicated, so nothing can drift.

    That scaling is also the correctness argument. The equation is never divided by the mass, so a
    ZERO mass row survives it: a Dirichlet row reduces to ``u^{n+1}[d] = g`` (unconditionally --
    cleaner than the theta-step's ``(g - (1-theta) u[d])/theta``), and a PRESSURE row of a saddle
    system, whose mass block is structurally empty, reduces to the incompressibility constraint
    evaluated at the new time.

    **Startup** is one backward-Euler step taken outside the scan, so the scan keeps a static trip
    count and no branch in its checkpointed body. BE's local error is O(dt^2), which is what a
    second-order global rate needs.

    Reference: Curtiss & Hirschfelder, *PNAS* **38** (1952) 235; Hairer & Wanner, *Solving Ordinary
    Differential Equations II*, 2nd ed., Sec. V.1.
    """

    step_order = 2

    def step_scales(self, block):
        """The coefficients of ``A`` in every step operator this scheme forms: the full-``dt`` startup
        step and the ``2dt/3`` BDF2 steps. Read at compose time so both are built eagerly."""
        dt = float(block.dt)
        return (dt, 2.0 * dt / 3.0)

    def adaptive(self, *, rtol: float = 1e-4, atol: float = 1e-6, max_steps: int = 1000, dt0: float | None = None):
        """BDF2 on error-controlled variable steps: :func:`jno.solve.bdf` at ``order=2`` (plain BDF, no NDF
        term, so the formula is the one this fixed-grid scheme takes)."""
        return _BDFScheme(2, ndf=False).adaptive(rtol=rtol, atol=atol, max_steps=max_steps, dt0=dt0)

    def integrate(self, block, args, save_ts, *, linear_solve=None, nonlinear_solve=None):
        import jax
        import jax.numpy as jnp

        from .backend_blocks import _resample_trajectory, _scheme_sharded
        from .history_march import _TRANSIENT_ADVICE, _check_march_converged

        sharded = _scheme_sharded(self, block, args, save_ts, linear_solve, nonlinear_solve)
        if sharded is not None:  # a linear march, its operators split over the devices
            return sharded
        md = block.metadata or {}
        # A STATE-DEPENDENT mass `c(u)*u_t` needs nothing extra here. Its mass action is assembled as
        # `c(u)(u - u_prev)/h` with `u_prev` the step's own starting state (see `SemidiscreteTimeBlock.step`),
        # and the reduction below starts each step from `u* = (4u^n - u^{n-1})/3` over `h = 2dt/3` -- so it
        # lands on exactly `c(u^{n+1})(3u^{n+1} - 4u^n + u^{n-1})/(2dt)`: BDF2 in its NON-conservative form,
        # second order in time (measured, `tests/test_fem_bdf2_state_dependent_mass.py`). It used to be
        # refused on the reading that one previous state left nowhere for the second level to enter; the
        # shifted start state is where it enters. What this is not: the conservative `(3H(u^{n+1}) - ...)`
        # form for an enthalpy-type mass `H(u)_t`, which would need `H` itself rather than `c = H'`.
        if md.get("second_order"):
            raise NotImplementedError(
                "jno.solve.bdf2(): a second-order-in-time (u_tt) block is assembled with theta=1/2 "
                "(trapezoidal) precisely because it must NOT be damped, and BDF2 is L-stable -- it "
                "would damp an undamped wave silently. Use the block's own scheme, or "
                "jno.solve.theta(0.5) explicitly."
            )

        s0 = block.state0_fn(args) if getattr(block, "state0_fn", None) is not None else block.state0
        s0 = jnp.asarray(s0).reshape(-1)
        dtype = s0.dtype
        dt = float(block.dt)
        # The grid is built on the HOST. `_block_time_grid` returns a `jnp.linspace`, and under a trace --
        # which is how `jno.core(...)` evaluates a solve, i.e. every inverse problem -- even a linspace of
        # plain floats is staged, so `float(grid_ts[1])` below raised a ConcretizationTypeError. BDF2 had
        # only ever been exercised eagerly (`.fn()`), so it was unusable inside `jno.core` without a word.
        import numpy as np

        t0, t1 = float(block.t0), float(block.t1)
        grid_ts = np.linspace(t0, t1, max(1, round((t1 - t0) / dt)) + 1)  # concrete: numpy, not jnp

        # A nonlinear step's own convergence guard cannot fire inside the scan, so the step reports
        # its residual norms and they are judged below -- as in the theta march and the load path.
        _judge = bool(block.is_nonlinear())

        from .backend_blocks import _split_cached_march, hoist_time_invariant

        dt_eff = 2.0 * dt / 3.0

        def march(s0, grid, args):
            blk = hoist_time_invariant(block, args, grid[0])  # static loads/operators: once per march

            def _advance(u_prev, t_land, h):
                """One implicit step landing at ``t_land`` from ``u_prev`` over an effective step ``h``."""
                return blk.step(
                    u_prev,
                    t_land - h,
                    h,
                    args=args,
                    theta=1.0,
                    linear_solve=linear_solve,
                    nonlinear_solve=nonlinear_solve,
                    report=_judge,
                )

            # Startup: plain backward Euler, then the BDF2 steps. Its norms are judged with the others.
            s1 = _advance(s0, grid[1], dt)

            def step(carry, t_next):
                u_n, u_nm1 = carry
                u_star = (4.0 * u_n - u_nm1) / 3.0
                out = _advance(u_star, t_next, dt_eff)
                if not _judge:
                    return (out, u_n), out
                wn, r_end, r_start = out
                return (wn, u_n), (wn, r_end, r_start)

            # Checkpointed for the same reason the theta march is: reverse mode would otherwise keep every
            # step's internal Krylov/Newton residuals.
            first = s1[0] if _judge else s1
            _, ys = jax.lax.scan(jax.checkpoint(step), (first, s0), grid[2:])
            return s1, ys

        from .backend_blocks import _march_eagerly, _march_to_host

        if _march_eagerly(args, save_ts):
            # Eager: the start-up step, then the BDF2 steps in chunks whose frames go to the host as they
            # are made -- the carry is the two-level state (u^n, u^{n-1}), so a chunk boundary changes
            # nothing about the scheme (see `_march_to_host`).
            def startup(s0, grid2, args):
                blk = hoist_time_invariant(block, args, grid2[0])
                return blk.step(
                    s0, grid2[1] - dt, dt, args=args, theta=1.0,
                    linear_solve=linear_solve, nonlinear_solve=nonlinear_solve, report=_judge,
                )  # fmt: skip

            from .backend_blocks import _carried_tangent0

            # The step tangent rides the carry (`SemidiscreteTimeBlock.step`, `tangent=`). It starts empty
            # rather than from the start-up step's: that one is scaled by 1/dt, the BDF2 steps by 3/(2dt).
            tang0 = _carried_tangent0(block, s0, nonlinear_solve, dtype)

            def make_step(args, t_start):
                blk = hoist_time_invariant(block, args, t_start)

                def step(carry, t_next):
                    (u_n, u_nm1), tang = carry if tang0 is not None else (carry, None)
                    u_star = (4.0 * u_n - u_nm1) / 3.0
                    out = blk.step(
                        u_star, t_next - dt_eff, dt_eff, args=args, theta=1.0,
                        linear_solve=linear_solve, nonlinear_solve=nonlinear_solve, report=_judge, tangent=tang,
                    )  # fmt: skip
                    if tang is not None:
                        out, tang = out
                    wn = out[0] if _judge else out
                    nxt = (wn, u_n) if tang is None else ((wn, u_n), tang)
                    return nxt, out

                return step

            cfg = (linear_solve, nonlinear_solve, "bdf2", dt)
            first = _split_cached_march(block, args, (*cfg, "startup"), startup, s0, jnp.asarray(grid_ts[:2], dtype), args)
            s1 = first[0] if _judge else first

            def _verdict(r_end, r_start, n_done, final, unchanged):
                _check_march_converged(
                    np.concatenate([np.reshape(np.asarray(first[1]), (1,)), r_end]),
                    np.concatenate([np.reshape(np.asarray(first[2]), (1,)), r_start]),
                    grid_ts[1 : n_done + 2],
                    nonlinear_solve,
                    what="transient march (BDF2)",
                    coord="t",
                    advice=_TRANSIENT_ADVICE,
                    unchanged=unchanged,
                )

            if grid_ts.size < 3:  # one step: the start-up step is the whole march
                if _judge:
                    _verdict(np.zeros((0,)), np.zeros((0,)), 0, True, None)
                traj = jnp.concatenate([s0[None, :], s1[None, :]], axis=0)
                return np.asarray(_resample_trajectory(traj, grid_ts, save_ts, dtype, self.step_order))
            return _march_to_host(
                block,
                args,
                (*cfg, tang0 is not None),
                make_step,
                (s1, s0) if tang0 is None else ((s1, s0), tang0),
                grid_ts[1:],
                dtype,
                save_ts,
                state_of=(lambda c: c[0]) if tang0 is None else (lambda c: c[0][0]),
                prefix_ts=grid_ts[:2],
                prefix_states=[s0, s1],
                judge=_verdict if _judge else None,
                degree=self.step_order,
            )

        # Traced ONCE per block and configuration (see `_cached_march`): an eager BDF2 march used to re-trace
        # its scan on every call -- measured ~0.3 s per warm call of a 12k-DOF heat march.
        s1, ys = _split_cached_march(
            block, args, (linear_solve, nonlinear_solve, "bdf2", dt), march, s0, jnp.asarray(grid_ts, dtype), args
        )
        if _judge:
            (s1, r1_end, r1_start), (ys, _r_end, _r_start) = s1, ys
            _check_march_converged(
                jnp.concatenate([jnp.reshape(r1_end, (1,)), _r_end]),
                jnp.concatenate([jnp.reshape(r1_start, (1,)), _r_start]),
                grid_ts[1:],
                nonlinear_solve,
                states=jnp.concatenate([s1[None, :], ys], axis=0),
                what="transient march (BDF2)",
                coord="t",
                advice=_TRANSIENT_ADVICE,
            )
        traj = jnp.concatenate([s0[None, :], s1[None, :], ys], axis=0)
        return _resample_trajectory(traj, grid_ts, save_ts, dtype, self.step_order)

    def __repr__(self):
        return "jno.solve.bdf2()"


def _sdirk_tableau(order):
    """Alexander's stiffly-accurate, L-stable SDIRK methods (SIAM J. Numer. Anal. 14(6), 1977, Table 1 / Sec. 2):
    ``(A, c, gamma)`` with ``b`` = the last row of ``A`` (stiff accuracy: the step's answer IS the last stage)."""
    import numpy as np

    if order == 2:
        g = 1.0 - 1.0 / np.sqrt(2.0)
        A = np.array([[g, 0.0], [1.0 - g, g]])
    elif order == 3:
        # gamma: the root in (1/6, 1/2) of x^3 - 3x^2 + 3x/2 - 1/6, which makes the method L-stable
        g = float(
            [r.real for r in np.roots([1.0, -3.0, 1.5, -1.0 / 6.0]) if abs(r.imag) < 1e-12 and 1 / 6 < r.real < 0.5][0]
        )
        tau = (1.0 + g) / 2.0
        b1 = -(6.0 * g * g - 16.0 * g + 1.0) / 4.0
        b2 = (6.0 * g * g - 20.0 * g + 5.0) / 4.0
        A = np.array([[g, 0.0, 0.0], [tau - g, g, 0.0], [b1, b2, g]])
    else:
        raise ValueError(f"jno.solve.sdirk(order={order!r}): order must be 2 or 3.")
    return A, A.sum(axis=1), float(g)


def _sdirk_shift(A):
    """``D`` with ``u*_i = u_n + sum_{k<i} D[i, k] Z_k`` (``Z_k = U_k - u_n``) -- the start state that turns stage
    ``i`` into one backward-Euler step of size ``gamma*dt``. From ``M Z = dt (A (x) I) F``: ``dt F_j = sum_k
    (A^-1)_jk M Z_k``, so the explicit part ``dt sum_{j<i} a_ij F_j = M sum_k D_ik Z_k`` needs no ``M^-1``."""
    import numpy as np

    Ainv = np.linalg.inv(A)
    s = A.shape[0]
    D = np.zeros_like(A)
    for i in range(s):
        for k in range(i):
            D[i, k] = sum(A[i, j] * Ainv[j, k] for j in range(k, i))
    return D


class _SDIRKScheme(_TimeScheme):
    """Singly diagonally implicit Runge-Kutta; see :func:`jno.solve.sdirk`.

    Implemented by REDUCTION, like BDF2: every stage is exactly one backward-Euler :meth:`step` of size
    ``gamma*dt`` from a shifted start state (``_sdirk_shift``), so the Dirichlet / zero-mass (DAE) rows, the
    solver slots, the Newton driver, the complex 2n path and the Krylov rescue all come from ``step``
    unchanged -- and all stages share ONE step operator ``M + gamma*dt*A``.
    """

    def __init__(self, order):
        import numpy as np

        self.order = int(order)
        self.A, self.c, self.gamma = _sdirk_tableau(self.order)
        self.D = _sdirk_shift(self.A)
        # Embedded solution u_hat = u + dt sum_j bhat_j F_j of order `order - 1`, from the SAME stages. With
        # M Z = dt (A x I) F it is u + sum_k d_k Z_k, d = bhat^T A^-1 -- no extra solve and no M^-1. Order 3:
        # bhat on stages 1-2 only, from sum bhat = 1 and bhat.c = 1/2; order 2: bhat = (1, 0). Their stability
        # functions are bounded at infinity (Rhat(inf) = -0.96 and -2.41), so a stiff mode inflates the
        # estimate only while it is present, not for the whole march.
        if self.order == 3:
            b2 = (1.0 - 2.0 * self.gamma) / (1.0 - self.gamma)
            bhat = np.array([1.0 - b2, b2, 0.0])
        else:
            bhat = np.array([1.0, 0.0])
        self.d_hat = bhat @ np.linalg.inv(self.A)
        self.embedded_order = self.order - 1

    @property
    def step_order(self):
        return self.order

    def step_scales(self, block):
        return (self.gamma * float(block.dt),)

    def _refuse(self, block):
        if (block.metadata or {}).get("second_order"):
            raise NotImplementedError(
                "jno.solve.sdirk(): a second-order-in-time (u_tt) block is assembled with theta=1/2 because it "
                "must NOT be damped, and this SDIRK is L-stable -- it would damp an undamped wave silently. "
                "Use the block's own scheme, or jno.solve.theta(0.5)."
            )

    def _one_step(self, blk, u, t, dt, args, linear_solve, nonlinear_solve, report):
        """One SDIRK step from ``u`` at ``t`` over ``dt``; returns ``u(t+dt)`` (and the stages' norms)."""
        s = self.A.shape[0]
        Z, reports = [], []
        for i in range(s):
            u_star = u
            for k in range(i):
                if self.D[i, k] != 0.0:
                    u_star = u_star + float(self.D[i, k]) * Z[k]
            h = self.gamma * dt
            out = blk.step(
                u_star,
                t + float(self.c[i]) * dt - h,
                h,
                args=args,
                theta=1.0,
                linear_solve=linear_solve,
                nonlinear_solve=nonlinear_solve,
                report=report is True,
            )
            if report is True:
                out, r_end, r_start = out
                reports.append((r_end, r_start))
            Z.append(out - u)
        if report == "embedded":
            u_hat = u
            for k, dk in enumerate(self.d_hat):
                if dk != 0.0:
                    u_hat = u_hat + float(dk) * Z[k]
            return u + Z[-1], u_hat
        return u + Z[-1], reports  # stiffly accurate: the last stage is the step's answer

    def stepper(self, block, args, *, linear_solve=None, nonlinear_solve=None):
        self._refuse(block)
        return lambda u, t, dt: self._one_step(block, u, t, dt, args, linear_solve, nonlinear_solve, False)[0]

    def embedded_stepper(self, block, args, *, linear_solve=None, nonlinear_solve=None):
        """``(u, t, dt) -> (u(t+dt), u_hat)``: the step and its embedded lower-order solution."""
        self._refuse(block)
        return lambda u, t, dt: self._one_step(block, u, t, dt, args, linear_solve, nonlinear_solve, "embedded")

    def integrate(self, block, args, save_ts, *, linear_solve=None, nonlinear_solve=None):
        import jax
        import jax.numpy as jnp

        from .backend_blocks import _resample_trajectory, _scheme_sharded, _split_cached_march, hoist_time_invariant
        from .matvec_format import prime

        self._refuse(block)
        sharded = _scheme_sharded(self, block, args, save_ts, linear_solve, nonlinear_solve)
        if sharded is not None:  # a linear march, its operators split over the devices
            return sharded
        prime(block.M, getattr(block, "A", None))
        _s0f = getattr(block, "state0_fn", None)
        s0 = jnp.asarray(_s0f(args) if _s0f is not None else block.state0).reshape(-1)
        dtype = s0.dtype
        import numpy as np

        dt = float(block.dt)
        # The grid on the HOST (numpy), as BDF2 does: under `jno.core` a jnp.linspace is staged, and the
        # per-stage convergence record below needs concrete times.
        t0, t1 = float(block.t0), float(block.t1)
        grid_np = np.linspace(t0, t1, max(1, round((t1 - t0) / dt)) + 1)
        grid_ts = jnp.asarray(grid_np, dtype)
        judge = bool(block.is_nonlinear())

        def make_step(args, t_start):
            blk = hoist_time_invariant(block, args, t_start)

            def step(u, t_next):
                un, reps = self._one_step(blk, u, t_next - dt, dt, args, linear_solve, nonlinear_solve, judge)
                if not judge:
                    return un, un
                return un, (un, jnp.stack([r[0] for r in reps]), jnp.stack([r[1] for r in reps]))

            return step

        def march(s0, grid_ts, args):
            return jax.lax.scan(jax.checkpoint(make_step(args, grid_ts[0])), s0, grid_ts[1:])

        from .backend_blocks import _march_eagerly, _march_to_host

        cfg = (linear_solve, nonlinear_solve, repr(self), dt)
        if _march_eagerly(args, save_ts):  # chunks, frames to the host as they are made (`_march_to_host`)
            from .history_march import _TRANSIENT_ADVICE, _check_march_converged

            n_stage = self.A.shape[0]

            def _verdict(r_end, r_start, n_done, final, unchanged):
                _check_march_converged(
                    r_end.reshape(-1),  # one entry per STAGE: every stage is a Newton solve that must converge
                    r_start.reshape(-1),
                    np.repeat(grid_np[1 : n_done + 1], n_stage),
                    nonlinear_solve,
                    what=f"transient march ({self!r}, per stage)",
                    coord="t",
                    advice=_TRANSIENT_ADVICE,
                )

            return _march_to_host(
                block, args, cfg, make_step, s0, grid_np, dtype, save_ts,
                state_of=lambda c: c, prefix_ts=grid_np[:1], prefix_states=[s0], judge=_verdict if judge else None,
                degree=self.step_order,
            )  # fmt: skip
        ys = _split_cached_march(block, args, cfg, lambda *a: march(*a)[1], s0, grid_ts, args)
        if judge:
            from .history_march import _TRANSIENT_ADVICE, _check_march_converged

            ys, r_end, r_start = ys
            s = self.A.shape[0]
            _check_march_converged(
                r_end.reshape(-1),  # one entry per STAGE: every stage is a Newton solve that must converge
                r_start.reshape(-1),
                np.repeat(grid_np[1:], s),
                nonlinear_solve,
                what=f"transient march ({self!r}, per stage)",
                coord="t",
                advice=_TRANSIENT_ADVICE,
            )
        traj = jnp.concatenate([s0[None, :], ys], axis=0)
        return _resample_trajectory(traj, grid_ts, save_ts, dtype, self.step_order)

    def __repr__(self):
        return f"jno.solve.sdirk(order={self.order})"


def _rosenbrock_tableau(method):
    """Standard-form Rosenbrock coefficients ``(a, g, b, order)`` (Hairer & Wanner II, Sec. IV.7, eq. (7.4)):
    ``a`` strictly lower (stage points), ``g`` lower with the constant diagonal ``gamma`` (Jacobian couplings).

    * ``ros34pw2`` -- Rang & Angermann, BIT Numer. Math. 45 (2005) 761-787: 4 stages, order 3, stiffly
      accurate (``b`` is the last row of ``a + g``), L-stable, a W-method (an approximate Jacobian keeps the
      order) and consistent for index-1 DAEs, i.e. algebraic (zero-mass) rows.
    * ``ros2`` -- 2 stages, order 2, L-stable with ``gamma = 1 + 1/sqrt(2)`` (Verwer, Spee, Blom & Hundsdorfer,
      SIAM J. Sci. Comput. 20(4), 1999): ``a21 = 1``, ``g21 = -2 gamma``, ``b = (1/2, 1/2)`` -- the standard-form
      solution of the order conditions ``sum b = 1``, ``b2 (a21 + g21) = 1/2 - gamma``. Not stiffly accurate.
    """
    import numpy as np

    if method == "ros34pw2":
        g0 = 4.3586652150845900e-01
        a = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [8.7173304301691801e-01, 0.0, 0.0, 0.0],
                [8.4457060015369423e-01, -1.1299064236484185e-01, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        g = np.array(
            [
                [g0, 0.0, 0.0, 0.0],
                [-8.7173304301691801e-01, g0, 0.0, 0.0],
                [-9.0338057013044082e-01, 5.4180672388095326e-02, g0, 0.0],
                [2.4212380706095346e-01, -1.2232505839045147e00, 5.4526025533510214e-01, g0],
            ]
        )
        b = np.array([2.4212380706095346e-01, -1.2232505839045147e00, 1.5452602553351020e00, 4.3586652150845900e-01])
        # the embedded order-2 solution of the same paper (Rhat(inf) = -0.48)
        bhat = np.array([3.7810903145819369e-01, -9.6042292212423178e-02, 0.5, 2.1793326075422950e-01])
        return a, g, b, 3, bhat
    if method == "ros2":
        gam = 1.0 + 1.0 / np.sqrt(2.0)
        return np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[gam, 0.0], [-2.0 * gam, gam]]), np.array([0.5, 0.5]), 2, None
    raise ValueError(f"jno.solve.rosenbrock(method={method!r}): method must be 'ros34pw2' or 'ros2'.")


class _RosenbrockScheme(_TimeScheme):
    """Linearly implicit Rosenbrock(-W) time scheme; see :func:`jno.solve.rosenbrock`.

    With ``R`` the block's residual (``M u' = -R(u, t)``) and ``J = dR/du`` at the step's start, every stage is
    ONE linear solve with the same matrix ``W = M + gamma h J``:

        ``W k_i = -h R(t_n + alpha_i h, u_n + sum_j a_ij k_j) - h J sum_{j<i} g_ij k_j - gamma_i h^2 R_t``,
        ``u_{n+1} = u_n + sum_i b_i k_i``,

    ``alpha_i = sum_j a_ij``, ``gamma_i = sum_{j<=i} g_ij`` -- no Newton loop at all.
    """

    needs_linear_step = True  # a nonlinear block still gets a LINEAR step solve (see compose_transient_step_solvers)

    def __init__(self, method):
        self.method = method
        self.a, self.g, self.b, self.order, self.bhat = _rosenbrock_tableau(method)
        if self.bhat is not None:
            self.embedded_order = self.order - 1
        self.gamma = float(self.g[0, 0])
        self.alpha = self.a.sum(axis=1)
        self.gsum = self.g.sum(axis=1)

    @property
    def step_order(self):
        return self.order

    def step_scales(self, block):
        return (self.gamma * float(block.dt),)

    def _refuse(self, block):
        md = block.metadata or {}
        if md.get("second_order"):
            raise NotImplementedError(
                f"{self!r}: a second-order-in-time (u_tt) block must NOT be damped, and this scheme is L-stable. "
                "Use the block's own scheme, or jno.solve.theta(0.5)."
            )
        if getattr(block, "mass_residual", None) is not None:
            raise NotImplementedError(
                f"{self!r}: a state-dependent mass c(u) u_t needs the dM/du term a linearly implicit scheme does "
                "not carry. Use jno.solve.sdirk() or bdf2(), which treat it by a Newton solve per stage."
            )

    def _pieces(self, blk, args):
        """``(R(u, t), J(u, t) -> BCOO | None, M(t))`` for a linear or a nonlinear block."""
        import jax.numpy as jnp

        from .linear import sparse_matvec

        if blk.is_nonlinear():
            return (
                lambda u, t: jnp.asarray(blk.residual(u, t, args)).reshape(-1),
                (lambda u, t: blk.jacobian(u, t, args)) if blk.jacobian is not None else None,
                lambda t: blk.mass(t, args),
            )

        def A_of(t):
            return blk.operator_fn(t, args) if blk.operator_fn is not None else blk.A

        def R(u, t):
            r = sparse_matvec(A_of(t))(u) if hasattr(A_of(t), "todense") else jnp.asarray(A_of(t)) @ u
            if blk.affine_bias is not None:
                r = r - jnp.asarray(blk.affine_bias, u.dtype).reshape(-1)
            if blk.forcing_vector_fn is not None:
                r = r - jnp.asarray(blk.forcing_vector_fn(t, args), u.dtype).reshape(-1)
            return r

        return R, (lambda u, t: A_of(t)), (lambda t: blk.mass_fn(t, args) if blk.mass_fn is not None else blk.M)

    def _one_step(self, blk, u, t, h, args, linear_solve, embedded=False):
        import jax
        import jax.numpy as jnp

        from .backend_blocks import _default_step_solve
        from .linear import matrix_diagonal, sparse_matvec
        from .solver_api import _add_step_operator

        R, jac, mass = self._pieces(blk, args)
        M = mass(t)
        J = jac(u, t) if jac is not None else None
        mv_M = sparse_matvec(M) if hasattr(M, "todense") else (lambda v: jnp.asarray(M) @ v)
        if J is not None and hasattr(J, "todense"):
            mv_J = sparse_matvec(J)
        elif J is not None:
            mv_J = lambda v: jnp.asarray(J) @ v  # noqa: E731
        else:  # matrix-free tangent: a JVP of the residual
            mv_J = lambda v: jax.jvp(lambda w: R(w, t), (u,), (v,))[1]  # noqa: E731
        s = h * self.gamma
        W_mv = lambda v: mv_M(v) + s * mv_J(v)  # noqa: E731
        diag = matrix_diagonal(M) + (s * matrix_diagonal(J) if J is not None else 0.0)
        R_t = jax.jvp(lambda tt: R(u, tt), (t,), (jnp.ones_like(t),))[1]
        W_op = None
        if linear_solve is not None and getattr(linear_solve, "wants_operator", False) and hasattr(J, "todense"):
            W_op = _add_step_operator(M, J, s) if hasattr(M, "todense") else None
        krylov = (blk.metadata or {}).get("krylov")

        def solve(rhs, x0):
            if linear_solve is None:
                return _default_step_solve(W_mv, rhs, x0, diag, krylov=krylov)
            kw = {}
            if getattr(linear_solve, "wants_scale", False):
                kw["scale"] = s
            if W_op is not None:
                kw["operator"] = W_op
            return linear_solve(W_mv, rhs, x0, lambda: diag, **kw)

        ks = []
        for i in range(self.a.shape[0]):
            Y = u
            coupled = jnp.zeros_like(u)
            for j in range(i):
                if self.a[i, j] != 0.0:
                    Y = Y + float(self.a[i, j]) * ks[j]
                if self.g[i, j] != 0.0:
                    coupled = coupled + float(self.g[i, j]) * ks[j]
            rhs = -h * R(Y, t + float(self.alpha[i]) * h) - float(self.gsum[i]) * h * h * R_t
            if i:
                rhs = rhs - h * mv_J(coupled)
            ks.append(solve(rhs, ks[-1] if ks else jnp.zeros_like(u)))
        out = u
        for i, k in enumerate(ks):
            out = out + float(self.b[i]) * k
        if embedded:
            u_hat = u
            for i, k in enumerate(ks):
                u_hat = u_hat + float(self.bhat[i]) * k
            return out, u_hat
        return out

    def stepper(self, block, args, *, linear_solve=None, nonlinear_solve=None):
        self._refuse(block)
        return lambda u, t, dt: self._one_step(block, u, t, dt, args, linear_solve)

    @property
    def embedded_stepper(self):
        """``(block, args, ...) -> ((u, t, dt) -> (u(t+dt), u_hat))`` where the method has an embedded pair
        (``ros34pw2``); absent otherwise, and the adaptive controller then doubles steps."""
        if self.bhat is None:
            raise AttributeError("embedded_stepper")

        def make(block, args, *, linear_solve=None, nonlinear_solve=None):
            self._refuse(block)
            return lambda u, t, dt: self._one_step(block, u, t, dt, args, linear_solve, embedded=True)

        return make

    def integrate(self, block, args, save_ts, *, linear_solve=None, nonlinear_solve=None):
        import jax
        import jax.numpy as jnp
        import numpy as np

        from .backend_blocks import _resample_trajectory, _scheme_sharded, _split_cached_march, hoist_time_invariant
        from .matvec_format import prime

        self._refuse(block)
        sharded = _scheme_sharded(self, block, args, save_ts, linear_solve, nonlinear_solve)
        if sharded is not None:  # a linear march, its operators split over the devices
            return sharded
        if nonlinear_solve is not None:
            raise ValueError(f"{self!r} is linearly implicit: there is no Newton solve for nonlinear= to drive.")
        prime(block.M, getattr(block, "A", None))
        _s0f = getattr(block, "state0_fn", None)
        s0 = jnp.asarray(_s0f(args) if _s0f is not None else block.state0).reshape(-1)
        dtype = s0.dtype
        dt = float(block.dt)
        t0, t1 = float(block.t0), float(block.t1)
        grid_ts = jnp.asarray(np.linspace(t0, t1, max(1, round((t1 - t0) / dt)) + 1), dtype)

        def make_step(args, t_start):
            blk = hoist_time_invariant(block, args, t_start)
            if block.is_nonlinear():
                if block.mass is not None and blk.mass is block.mass:
                    raise NotImplementedError(
                        f"{self!r}: the mass matrix depends on time, which a linearly implicit scheme would need an "
                        "M'(t) u term for. Use jno.solve.sdirk() or bdf2()."
                    )
            elif block.mass_fn is not None and blk.mass_fn is block.mass_fn:
                raise NotImplementedError(
                    f"{self!r}: the mass matrix depends on time (or on a parameter), which a linearly implicit "
                    "scheme would need an M'(t) u term for. Use jno.solve.sdirk() or bdf2()."
                )

            def step(u, t_next):
                un = self._one_step(blk, u, t_next - dt, dt, args, linear_solve)
                return un, un

            return step

        def march(s0, grid_ts, args):
            return jax.lax.scan(jax.checkpoint(make_step(args, grid_ts[0])), s0, grid_ts[1:])

        from .backend_blocks import _march_eagerly, _march_to_host

        cfg = (linear_solve, repr(self), dt)
        if _march_eagerly(args, save_ts):  # chunks, frames to the host as they are made (`_march_to_host`)
            grid_np = np.asarray(grid_ts, dtype=float)
            return _march_to_host(
                block, args, cfg, make_step, s0, grid_np, dtype, save_ts,
                state_of=lambda c: c, prefix_ts=grid_np[:1], prefix_states=[s0], degree=self.step_order,
            )  # fmt: skip
        ys = _split_cached_march(block, args, cfg, lambda *a: march(*a)[1], s0, grid_ts, args)
        traj = jnp.concatenate([s0[None, :], ys], axis=0)
        return _resample_trajectory(traj, grid_ts, save_ts, dtype, self.step_order)

    def __repr__(self):
        return f"jno.solve.rosenbrock(method={self.method!r})"


class _AdaptiveScheme(_TimeScheme):
    """Step-size **policy** wrapping a base scheme — step-doubling (Richardson) error control on that
    scheme's implicit step, so it inherits the block's DAE handling and works for a linear or nonlinear
    block, scalar or vector, plain or periodic-reduced (the step runs in the reduced space; the caller
    prolongs). The complex-transient path feeds the same marcher its 2n-block step.

    Built by :meth:`_TimeScheme.adaptive` (``jno.solve.theta(0.5).adaptive(...)``). ``base=None`` — what
    the bare :func:`jno.solve.adaptive` produces — means "whatever θ-step the assembly picked for this
    block", which is backward Euler for a parabolic block and trapezoidal for a second-order one."""

    def __init__(
        self, base, rtol: float, atol: float, max_steps: int, dt0: float | None = None, limit=None, shrink=0.5, grow=1.5
    ):
        self.base = base
        self.rtol, self.atol, self.max_steps = float(rtol), float(atol), int(max_steps)
        self.dt0 = None if dt0 is None else float(dt0)
        # `limit` switches the controller from Richardson local error (a TIME-accuracy notion, meaningless
        # on a rate-independent load path) to a bound on how much the solution may change in one step.
        # Only the `tau=` load-path slot reads it; `time=` rejects it. See `jno.solve.adaptive`.
        self.limit = limit
        self.shrink, self.grow = float(shrink), float(grow)

    def base_for(self, block):
        """The base scheme this will actually step with — ``self.base``, or the block's own θ-step."""
        if self.base is not None:
            return self.base
        theta = float(block.metadata.get("theta", 1.0)) if getattr(block, "metadata", None) else 1.0
        return _ThetaScheme(theta)

    def adaptive(self, **kwargs):
        raise NotImplementedError("this scheme is already adaptively sized; .adaptive() does not nest.")

    @staticmethod
    def _embedded_of(base):
        """The base scheme's embedded-pair stepper factory, or ``None`` (then the controller doubles steps)."""
        return getattr(base, "embedded_stepper", None)

    def integrate(self, block, args, save_ts, *, linear_solve=None, nonlinear_solve=None):
        if self.limit is not None:
            raise ValueError(
                "fem.solve(time=jno.solve.adaptive(limit=...)): `limit` bounds the per-step SOLUTION CHANGE "
                "on a pseudo-time load path (`fem.solve(tau=...)` over a `domain(tau=...)` march). A "
                "transient is sized by the local truncation error instead — drop `limit` and use "
                "rtol/atol, or move the spec to the tau= slot."
            )
        s0f = getattr(block, "state0_fn", None)
        u0 = jnp.asarray(s0f(args) if s0f is not None else block.state0).reshape(-1)
        base = self.base_for(block)
        step_fn = base.stepper(block, args, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve)
        # one step per attempt where the scheme has an embedded pair (sdirk, ros34pw2), else step doubling
        make_embedded = self._embedded_of(base)
        embedded = (
            make_embedded(block, args, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve)
            if make_embedded is not None
            else None
        )

        # NOTE: deliberately NOT ``block.dt``. The output grid's dt has no relation to the correct step
        # size; an over-large first step is rejected and retried, which costs attempts from the budget,
        # while ``dt0=None`` starts at the floor and grows into the right scale. See ``adaptive_march``.
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
            order=base.embedded_order if embedded is not None else base.step_order,
            dense_degree=base.step_order,
            embedded=embedded,
            error_rows=_error_rows(block, args, u0),
        )

    def __repr__(self):
        base = "" if self.base is None else f"{self.base!r}."
        return f"{base}adaptive(rtol={self.rtol}, atol={self.atol}, max_steps={self.max_steps}, dt0={self.dt0})"


# Klopfenstein-Shampine NDF constants kappa_k, k = 1..5 (Shampine & Reichelt, SIAM J. Sci. Comput. 18 (1997),
# Table 1; NDF5 is BDF5). Index 0 is unused.
_NDF_KAPPA = (0.0, -0.1850, -1.0 / 9.0, -0.0823, -0.0415, 0.0)
_BDF_MAX_ORDER = 5


class _BDFScheme(_TimeScheme):
    """Variable-step, variable-order BDF / NDF; see :func:`jno.solve.bdf`.

    Each step solves ``sum_j alpha_j u^{n+1-j} - (kappa*gamma_k/h)(u^{n+1} - P(t_{n+1})) = f(t_{n+1}, u^{n+1})``,
    ``alpha_j`` the derivative at ``t_{n+1}`` of the Lagrange basis through the ACTUAL past step times
    (variable-coefficient BDF), ``P`` the predictor through the ``k + 1`` past states and ``kappa`` the NDF
    constant (0 = plain BDF). It is linear in ``u^{n+1}``, so -- as for :class:`_BDF2Scheme` -- it is EXACTLY a
    backward-Euler step of size ``1/alpha_0'`` from a shifted state ``psi``, and reuses the block's step
    unchanged: Dirichlet / algebraic rows, Newton, the solver slots and the implicit adjoint.

    ``tol`` is ``(rtol, atol, max_steps, dt0)`` for the error-controlled march (:func:`bdf_march`), or
    ``None`` for the domain's fixed grid, offered for ``order`` 1 (backward Euler) and 2 (BDF2) only."""

    def __init__(self, order=None, ndf=True, tol=None):
        self.order = order
        self.ndf = bool(ndf)
        self.tol = tol

    @property
    def step_order(self):
        return _BDF_MAX_ORDER if self.order is None else int(self.order)

    def _fixed(self):
        """The fixed-grid scheme this stands for, or ``None`` when the march is error-controlled."""
        if self.tol is not None:
            return None
        return _ThetaScheme(1.0) if self.order == 1 else _BDF2Scheme()

    def step_scales(self, block):
        fixed = self._fixed()
        return fixed.step_scales(block) if fixed is not None else ()  # a traced step: nothing to pre-build

    def adaptive(self, *, rtol: float = 1e-4, atol: float = 1e-6, max_steps: int = 1000, dt0: float | None = None):
        """Set the tolerances and the step budget of the error-controlled march (see :func:`jno.solve.bdf`)."""
        return _BDFScheme(self.order, self.ndf, (float(rtol), float(atol), int(max_steps), dt0))

    def integrate(self, block, args, save_ts, *, linear_solve=None, nonlinear_solve=None):
        fixed = self._fixed()
        if fixed is not None:
            return fixed.integrate(block, args, save_ts, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve)
        if (block.metadata or {}).get("second_order"):
            raise NotImplementedError(
                f"{self!r}: a second-order-in-time (u_tt) block is assembled with theta=1/2 because it must NOT "
                "be damped, and BDF is L-stable at orders 1-2 and strongly damping above -- it would damp an "
                "undamped wave silently. Use the block's own scheme, or jno.solve.theta(0.5)."
            )
        rtol, atol, max_steps, dt0 = self.tol
        s0f = getattr(block, "state0_fn", None)
        u0 = jnp.asarray(s0f(args) if s0f is not None else block.state0).reshape(-1)
        be = _ThetaScheme(1.0).stepper(block, args, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve)
        return bdf_march(
            be,
            u0,
            float(block.t0),
            float(block.t1),
            save_ts,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            dt0=dt0,
            max_order=self.step_order,
            vary_order=self.order is None,
            ndf=self.ndf,
            error_rows=_error_rows(block, args, u0),
        )

    def __repr__(self):
        tol = "" if self.tol is None else f".adaptive(rtol={self.tol[0]}, atol={self.tol[1]}, max_steps={self.tol[2]})"
        return f"jno.solve.bdf(order={self.order}, ndf={self.ndf}){tol}"


def _lagrange_derivative_at_first(tau):
    """``l_j'(tau_0)`` for the Lagrange basis on the nodes ``tau = (tau_0, ..., tau_k)`` (distinct)."""
    k1 = len(tau)
    out = []
    for j in range(k1):
        if j == 0:
            out.append(sum(1.0 / (tau[0] - tau[m]) for m in range(1, k1)))
            continue
        num = 1.0
        for m in range(1, k1):
            if m != j:
                num = num * (tau[0] - tau[m])
        den = 1.0
        for m in range(k1):
            if m != j:
                den = den * (tau[j] - tau[m])
        out.append(num / den)
    return out


def _extrapolation_weights(x, tau):
    """Lagrange basis weights of the nodes ``tau`` at ``x`` (the predictor through past states)."""
    out = []
    for j in range(len(tau)):
        w = 1.0
        for m in range(len(tau)):
            if m != j:
                w = w * (x - tau[m]) / (tau[j] - tau[m])
        out.append(w)
    return out


def bdf_march(
    be_step,
    u0,
    t0,
    t1,
    save_ts,
    *,
    rtol,
    atol,
    max_steps,
    dt0=None,
    max_order=_BDF_MAX_ORDER,
    vary_order=True,
    ndf=True,
    error_rows=None,
    safety=0.9,
    min_factor=0.2,
    max_factor=2.0,
    deadband=1.2,
):
    """Error-controlled variable-step, variable-order BDF / NDF march, reverse-mode differentiable.

    ``be_step(w, t, dt, guess=None)`` is the block's backward-Euler step ``(M + dt J) u = M w + dt f(t + dt)``. A BDF step
    of order ``k`` to ``t_{n+1} = t_n + h`` is ``be_step(psi, t_{n+1} - g, g)`` with ``g = 1/alpha_0'`` and
    ``psi`` the combination of the past states the formula fixes (see :class:`_BDFScheme`).

    **Error.** The local error of order ``k`` is ``(kappa_k gamma_k + 1/(k+1)) (u^{n+1} - P_k(t_{n+1}))``,
    ``P_k`` the predictor through the ``k + 1`` past states (the ``k+1``-th backward difference on equal
    steps; Shampine & Reichelt 1997, Sec. 2-3). Its RMS in the mixed tolerance over ``error_rows`` decides:
    above 1 the attempt is REJECTED (state kept, step shrunk, no growth on the next attempt), as in
    :func:`adaptive_march`, whose differentiability argument this shares (zero cotangent for a rejected
    state, the controller ``stop_gradient``-ed).

    **Order** (``vary_order``): after ``k + 1`` steps at order ``k``, the errors the orders ``k - 1`` and
    ``k + 1`` would have made are estimated the same way (predictors through ``k`` and ``k + 2`` past states)
    and the order allowing the largest next step is taken, ``k +- 1`` handicapped by ``1.2``. The march
    starts at order 1 from the floor step and grows from there, so no starting procedure is needed.

    **Step.** Factor ``safety * err^(-1/(k+1))`` clipped to ``[min_factor, max_factor]``; a factor in
    ``[1, deadband]`` keeps the step (variable-coefficient BDF is zero-stable under bounded step ratios, and an
    unchanged step keeps the step operator). Fixed-length ``lax.scan`` of ``max_steps`` attempts; an exhausted
    budget NaN-poisons the result. Output at ``save_ts``: degree-``max_order`` interpolation through the
    accepted states (:func:`_dense_sample`)."""
    u0 = jnp.asarray(u0).reshape(-1)
    rt = jnp.real(u0).dtype
    t0, t1 = jnp.asarray(t0, rt), jnp.asarray(t1, rt)
    span = jnp.abs(t1 - t0)
    dt_min, dt_max = 1e-4 * span, span
    dt_start = dt_min if dt0 is None else jnp.clip(jnp.asarray(dt0, rt), dt_min, dt_max)
    K = int(max_order)
    if not 1 <= K <= _BDF_MAX_ORDER:
        raise ValueError(f"BDF order must be in 1..{_BDF_MAX_ORDER}; got {K}.")
    H = K + 2  # history slots: the current state and K + 1 before it (the order-(K+1) estimate needs K + 2)
    kap = [(_NDF_KAPPA[k] if ndf and k < len(_NDF_KAPPA) else 0.0) for k in range(K + 2)]  # order K+1: an estimate only
    gam = [sum(1.0 / m for m in range(1, k + 1)) for k in range(K + 3)]
    err_c = [0.0] + [kap[k] * gam[k] + 1.0 / (k + 1) for k in range(1, K + 2)]

    weight = None
    if error_rows is not None:
        weight = jax.lax.stop_gradient(jnp.asarray(error_rows, rt).reshape(-1))
        n_counted = jnp.maximum(jnp.sum(weight), 1.0)

    def _rms(e, a, b):
        a, b, e = jax.lax.stop_gradient(a), jax.lax.stop_gradient(b), jax.lax.stop_gradient(e)
        r2 = jnp.abs(e / (atol + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b)))) ** 2
        if weight is None:
            return jnp.sqrt(jnp.mean(r2))
        return jnp.sqrt(jnp.sum(weight * r2) / n_counted)

    def _formula(k, t_new, h, hist_t, n_hist):
        """Row weights over the history (``H`` slots) of ``psi``, the step ``g``, and whether the predictor
        exists, for the static order ``k``."""
        tau = [t_new] + [hist_t[j] for j in range(k)]
        a = _lagrange_derivative_at_first(tau)
        have_pred = n_hist >= k + 1
        kp = jnp.where(have_pred, kap[k], 0.0) * gam[k] / h  # NDF term, plain BDF until the predictor exists
        beta = _extrapolation_weights(t_new, [hist_t[j] for j in range(k + 1)])
        a0 = a[0] - kp
        c = [0.0] * H
        for j in range(1, k + 1):
            c[j - 1] = c[j - 1] - a[j] / a0
        for j in range(k + 1):
            c[j] = c[j] - kp * beta[j] / a0
        return jnp.stack([jnp.asarray(x, rt) for x in c]), 1.0 / a0

    def _predictor(q, t_new, hist_t, hist_u):
        """``P_q(t_new)``, the polynomial through the ``q + 1`` newest past states."""
        beta = _extrapolation_weights(t_new, [hist_t[j] for j in range(q + 1)])
        return sum(beta[j] * hist_u[j] for j in range(q + 1))

    def _pred_gap(q, u_new, t_new, hist_t, hist_u):
        """``u_new - P_q(t_new)``."""
        return u_new - _predictor(q, t_new, hist_t, hist_u)

    def _attempt(operand):
        k, t, h, hist_t, hist_u, n_hist = operand
        t_new = t + h

        def _one(kk):
            c, g = _formula(kk, t_new, h, hist_t, n_hist)
            psi = jnp.tensordot(c, hist_u, axes=1)
            # The step's solve starts from the predictor, O(h^(k+1)) from the answer, not from psi, which is
            # O(h) from it (CVODE / ode15s start Newton there too); before the predictor exists, from u^n.
            pred = _predictor(kk, t_new, hist_t, hist_u)
            u_new = be_step(psi, t_new - g, g, guess=jnp.where(n_hist >= kk + 1, pred, hist_u[0]))
            gap = u_new - pred
            err = jnp.where(n_hist >= kk + 1, _rms(err_c[kk] * gap, u_new, hist_u[0]), 0.0)
            # the neighbouring orders' errors, for the order choice (inf where they cannot be estimated)
            lo = (
                _rms(err_c[kk - 1] * _pred_gap(kk - 1, u_new, t_new, hist_t, hist_u), u_new, hist_u[0])
                if kk > 1
                else jnp.inf
            )
            hi = (
                jnp.where(
                    n_hist >= kk + 2,
                    _rms(err_c[kk + 1] * _pred_gap(kk + 1, u_new, t_new, hist_t, hist_u), u_new, hist_u[0]),
                    jnp.inf,
                )
                if kk < K
                else jnp.asarray(jnp.inf, rt)
            )
            return u_new, err, jnp.asarray(lo, rt), jnp.asarray(hi, rt)

        return jax.lax.switch(k - 1, [lambda kk=kk: _one(kk) for kk in range(1, K + 1)])

    def body(carry, _):
        t, hist_t, hist_u, n_hist, k, n_at_k, dt, grow_ok = carry
        u = hist_u[0]
        n_prev = n_hist  # past states this attempt's formula and estimate could read
        remaining = t1 - t
        done = remaining <= dt_min
        dt_step = jnp.clip(jnp.minimum(jnp.clip(dt, dt_min, dt_max), remaining), dt_min, dt_max)
        _t, _h = jax.lax.stop_gradient(t), jax.lax.stop_gradient(dt_step)
        _ht = jax.lax.stop_gradient(hist_t)
        skip = (u, jnp.asarray(0.0, rt), jnp.asarray(jnp.inf, rt), jnp.asarray(jnp.inf, rt))
        u_try, err, err_lo, err_hi = jax.lax.cond(done, lambda o: skip, _attempt, (k, _t, _h, _ht, hist_u, n_hist))
        at_floor = _h <= dt_min * (1.0 + 1e-9)
        accept = done | (err <= 1.0) | at_floor
        moved = accept & ~done
        # history: the new state in front on an accepted step
        hist_u = jnp.where(moved, jnp.concatenate([u_try[None, :], hist_u[:-1]], axis=0), hist_u)
        hist_t = jnp.where(moved, jnp.concatenate([(t + _h)[None], hist_t[:-1]]), hist_t)
        n_hist = jnp.where(moved, jnp.minimum(n_hist + 1, H), n_hist)
        t_new = jnp.where(moved, t + _h, t)

        def fac_of(e, q):
            return jnp.clip(safety * jnp.maximum(e, 1e-12) ** (-1.0 / (q + 1)), min_factor, max_factor)

        f_k = jnp.where(n_prev >= k + 1, fac_of(err, k), max_factor)  # before the estimate exists: grow
        k_new, fac = k, f_k
        if vary_order:
            ready = moved & (n_at_k >= k + 1)
            f_lo = jnp.where(ready & (k > 1), fac_of(err_lo, k - 1) / 1.2, 0.0)
            f_hi = jnp.where(ready & (k < K) & (n_prev >= k + 2), fac_of(err_hi, k + 1) / 1.2, 0.0)
            down, up = f_lo > jnp.maximum(f_k, f_hi), f_hi > jnp.maximum(f_k, f_lo)
            k_new = jnp.where(down, k - 1, jnp.where(up, k + 1, k))
            fac = jnp.where(down, f_lo, jnp.where(up, f_hi, f_k))
        fac = jnp.where(grow_ok & accept, fac, jnp.minimum(fac, 1.0))
        fac = jnp.where((fac >= 1.0) & (fac <= deadband), 1.0, fac)
        if not vary_order:  # fixed order: climb to it as the history fills
            k_new = jnp.where(moved & (n_hist >= k + 1) & (k < K), k + 1, k)
        dt_next = jnp.clip(_h * fac, dt_min, dt_max)
        dt_next = jnp.where(accept & (_h < dt), jnp.maximum(dt_next, dt), dt_next)  # a step cut to land on t1
        n_at_k = jnp.where(k_new != k, 0, jnp.where(moved, n_at_k + 1, n_at_k))
        new = (t_new, hist_t, hist_u, n_hist, k_new.astype(jnp.int32), n_at_k, dt_next, accept)
        return new, (t_new, hist_u[0])

    hist_t0 = t0 - span * jnp.arange(H, dtype=rt)  # distinct placeholders before t0 (never read: n_hist gates them)
    hist_u0 = jnp.broadcast_to(u0, (H, u0.shape[0]))
    init = (t0, hist_t0, hist_u0, jnp.asarray(1, jnp.int32), jnp.asarray(1, jnp.int32), jnp.asarray(0, jnp.int32),
            dt_start, jnp.asarray(True))  # fmt: skip
    (t_end, *_), (ts, us) = jax.lax.scan(body, init, None, length=int(max_steps))
    ts = jax.lax.stop_gradient(jnp.concatenate([t0[None], ts]))
    us = jnp.concatenate([u0[None, :], us], axis=0)
    out = _dense_sample(jnp.asarray(save_ts, rt), ts, us, K)
    reached = t_end >= t1 - 2.0 * dt_min
    return out * jnp.where(reached, jnp.asarray(1.0, rt), jnp.asarray(jnp.nan, rt))


def _error_rows(block, args, u0):
    """``1`` on the rows the adaptive error norm counts, ``0`` on the rows it leaves out, or ``None`` (count
    every row).

    Left out: algebraic rows (zero mass) whose own diagonal of the operator is zero, so the row does not
    determine its own unknown: the incompressibility row of a saddle problem, whose pressure is an
    index-2 variable. Its local error converges one order lower than the differential rows', so counting
    it shrinks the steps without bound (Hairer & Wanner, *Solving ODEs II*, Sec. VII.3 and VII.4; SUNDIALS
    IDA leaves such rows out with ``IDASetSuppressAlg``). Kept: an algebraic row that fixes its own unknown
    (a Dirichlet row, a pressure pin, a stabilised pressure row), whose error is controlled like a state.
    ``None`` when the block carries no mass matrix of the state's size (a state-dependent mass residual)."""
    from .backend_blocks import _block_index2_rows

    leave_out = _block_index2_rows(block, args, u0)
    if leave_out is None or leave_out.shape[0] != u0.shape[0]:
        return None
    return (~leave_out).astype(jnp.real(u0).dtype)


def _theta_order(theta: float) -> int:
    """Classical order of the θ-step: the θ-method is O(dt²) **only** at θ=1/2 (Crank–Nicolson /
    trapezoidal) and O(dt) everywhere else. The step-size controller needs this to pick its exponent
    ``1/(p+1)``; getting it wrong does not break the march, it just sizes every step badly."""
    return 2 if abs(float(theta) - 0.5) < 1e-12 else 1


def _dense_sample(save, ts, us, degree):
    """Sample a march's stored states ``us`` (at the nondecreasing times ``ts``) at ``save``, by piecewise
    Lagrange interpolation of ``degree`` through neighbouring DISTINCT step times (Hairer, Norsett & Wanner I,
    Sec. II.6: a continuous extension of the method's order needs an interpolant of that order).

    ``degree = 1`` is the linear blend; a scheme of order ``p`` passes ``p``, whose interpolation error
    ``O(h^(p+1))`` stays below the step's own ``O(h^p)``. A rejected or settled adaptive attempt records a
    DUPLICATE time; each distinct time is read at its first index, so no node repeats and no weight divides by
    a zero gap. The weights depend only on the (``stop_gradient``-ed) times: the result is linear in ``us`` and
    differentiable through it. Fewer distinct times than ``degree + 1`` fall back to the linear blend. Times
    outside ``[ts[0], ts[-1]]`` are clamped to the end states, as before. The fixed-grid paths use the same
    weights (:func:`jno.utils.solver.backend_blocks._resample_trajectory`)."""
    from .backend_blocks import _lagrange_weights

    ts = jax.lax.stop_gradient(ts)
    m = ts.shape[0]
    p = max(1, int(degree))
    # rank of every stored time among the distinct ones; first stored index of every rank
    rank = jnp.concatenate([jnp.zeros((1,), jnp.int32), jnp.cumsum((ts[1:] > ts[:-1]).astype(jnp.int32))])
    n_distinct = rank[-1] + 1
    first = jnp.searchsorted(rank, jnp.arange(m, dtype=jnp.int32), side="left")  # rank -> index (valid below n_distinct)
    x = jnp.clip(save, ts[0], ts[-1])
    left = rank[jnp.clip(jnp.searchsorted(ts, x, side="right") - 1, 0, m - 1)]  # rank of the left bracket

    def _sample(q):
        """Degree-``q`` interpolant on the ``q + 1`` ranks around each save time (start clipped into range)."""
        start = jnp.clip(left - (q - 1) // 2, 0, jnp.maximum(n_distinct - 1 - q, 0))
        idx = first[jnp.clip(start[:, None] + jnp.arange(q + 1)[None, :], 0, n_distinct - 1)]  # (S, q+1)
        out = 0.0
        for j, wj in enumerate(_lagrange_weights(ts[idx], x)):
            out = out + wj[:, None] * us[idx[:, j]]
        return out

    if p == 1:
        return _sample(1)
    # too few distinct times for degree p (a march of one or two steps): the linear blend
    return jnp.where(n_distinct > p, _sample(p), _sample(1))


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
    dense_degree=1,
    embedded=None,
    error_rows=None,
    safety=0.9,
    min_factor=0.2,
    max_factor=5.0,
):
    """Adaptive step-size march with error control and step REJECTION, reverse-mode differentiable.

    Two error estimates:

    * **embedded** (``embedded(u, t, dt) -> (u_new, u_hat)``): the base scheme's own lower-order solution
      ``u_hat`` from the SAME stages (Hairer & Wanner II, Sec. IV.8; Rosenbrock: Rang & Angermann 2005). One
      step per attempt; ``order`` is then the order ``q`` of ``u_hat``, whose local error ``O(dt^(q+1))`` the
      controller sizes with exponent ``1/(q+1)``.
    * **step doubling** (no ``embedded``): ``step_fn(u, t, dt) -> u(t+dt)`` once over ``dt`` and twice over
      ``dt/2``, three steps per attempt (Hairer, Norsett & Wanner I, Sec. II.4); ``order`` is the step's order
      ``p`` and the exponent ``1/(p+1)``. Kept for the theta-steps, which have no embedded pair.

    The error is the RMS of ``(u_new - u_hat) / (atol + rtol * max(|u_new|, |u_hat|))`` over the rows
    ``error_rows`` marks (all rows when ``None``; the caller leaves out index-2 rows such as the
    incompressibility row, see ``_error_rows``). An attempt with error
    above 1 is REJECTED -- the state and time stay, the step shrinks -- unless it is already at the floor
    ``dt_min = 1e-4 * span``; after a rejection the next step may not grow. The march is a fixed-length
    ``lax.scan`` of ``max_steps`` attempts (rejections count) -- a static trip count, so it stays reverse
    differentiable -- where the settled tail consumes attempts without solving. The trajectory is sampled at
    ``save_ts`` by a degree-``dense_degree`` interpolant through the accepted states (:func:`_dense_sample`;
    the caller passes the base step's order, so output between steps keeps it); an exhausted budget
    NaN-poisons the result (raise ``max_steps``).

    Rejection is differentiable because a discarded attempt's states receive an exactly ZERO cotangent, and
    the step solves return 0 for a zero right-hand side (``backend_blocks._default_step_solve``) instead of
    the 0/0 = NaN JAX's BiCGStab gives there. The controller (error, accept, next ``dt``) is ``stop_gradient``-
    ed, as in Diffrax: the gradient is the derivative at the realized step schedule."""
    u0 = jnp.asarray(u0).reshape(-1)
    rt = jnp.real(u0).dtype  # time / dt live in the state's real dtype (a complex system marches its real 2n block)
    t0, t1 = jnp.asarray(t0, rt), jnp.asarray(t1, rt)
    span = jnp.abs(t1 - t0)
    # A non-degenerate floor: a *tiny* dt makes ``M + dt·A`` near-singular on the zeroed Dirichlet rows
    # (its adjoint then blows up to NaN), so never step below this.
    dt_min = 1e-4 * span
    dt_max = span
    # With rejection an over-large first step is retried, not kept; starting at the floor still costs only
    # the few attempts it takes to grow (at most `max_factor` each) and never one wasted rejection.
    dt_start = dt_min if dt0 is None else jnp.clip(jnp.asarray(dt0, rt), dt_min, dt_max)

    # ``error_rows`` (1 counted, 0 left out; see ``_error_rows``): the mean runs over the counted rows, so the
    # tolerance keeps its meaning.
    weight = None
    if error_rows is not None:
        weight = jax.lax.stop_gradient(jnp.asarray(error_rows, rt).reshape(-1))
        n_counted = jnp.maximum(jnp.sum(weight), 1.0)

    def _err(a, b):  # normalized RMS of the two estimates' difference (mixed absolute / relative tolerance)
        a, b = jax.lax.stop_gradient(a), jax.lax.stop_gradient(b)
        scale = atol + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b))
        r2 = jnp.abs((a - b) / scale) ** 2
        if weight is None:
            return jnp.sqrt(jnp.mean(r2))
        return jnp.sqrt(jnp.sum(weight * r2) / n_counted)

    def _take(operand):
        u, _t, _dt = operand
        if embedded is not None:
            u_new, u_hat = embedded(u, _t, _dt)
            return u_new, _err(u_new, u_hat)
        # ``u_full`` is used ONLY to size the next step, so freeze its gradient entirely.
        u_full = jax.lax.stop_gradient(step_fn(u, _t, _dt))
        u_half = step_fn(step_fn(u, _t, 0.5 * _dt), _t + 0.5 * _dt, 0.5 * _dt)  # the kept (more accurate) state
        return u_half, _err(u_full, u_half)

    k = int(order) + 1  # the estimate's local error is O(dt^k)

    def body(carry, _):
        t, u, dt, grow_ok = carry  # t, dt are the step SCHEDULE (control); the gradient flows through u only
        remaining = t1 - t
        done = remaining <= dt_min
        dt_ctrl = jnp.clip(dt, dt_min, dt_max)
        dt_step = jnp.clip(jnp.minimum(dt_ctrl, remaining), dt_min, dt_max)  # land on t1; never a degenerate dt
        # Differentiate the numerical state at the REALIZED (t, dt) schedule; the controller is a discrete
        # control decision, `stop_gradient`-ed (Diffrax does the same). On a settled step `lax.cond` runs the
        # skip branch (no solve).
        _t, _dt = jax.lax.stop_gradient(t), jax.lax.stop_gradient(dt_step)
        u_try, err = jax.lax.cond(done, lambda o: (o[0], jnp.asarray(0.0, rt)), _take, (u, _t, _dt))
        at_floor = _dt <= dt_min * (1.0 + 1e-9)
        accept = done | (err <= 1.0) | at_floor
        u_new = jnp.where(accept, u_try, u)  # a rejected attempt's state gets a zero cotangent (see docstring)
        t_new = jnp.where(done | ~accept, t, t + _dt)
        fac = jnp.clip(safety * jnp.maximum(err, 1e-12) ** (-1.0 / k), min_factor, max_factor)
        fac = jnp.where(grow_ok, fac, jnp.minimum(fac, 1.0))  # no growth straight after a rejection
        dt_next = jnp.clip(dt_step * fac, dt_min, dt_max)
        # an accepted step that was shortened to land on t1 keeps the controller's size for what follows
        dt_next = jnp.where(accept & (dt_step < dt_ctrl), jnp.maximum(dt_next, dt_ctrl), dt_next)
        return (t_new, u_new, dt_next, accept), (t_new, u_new)

    init = (t0, u0, dt_start, jnp.asarray(True))
    (t_end, *_), (ts, us) = jax.lax.scan(body, init, None, length=int(max_steps))
    ts = jax.lax.stop_gradient(jnp.concatenate([t0[None], ts]))  # sample times are a fixed schedule (control)
    us = jnp.concatenate([u0[None, :], us], axis=0)
    save = jnp.asarray(save_ts, rt)
    out = _dense_sample(save, ts, us, dense_degree)
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


def _refuse_what_the_mask_would_zero(Aop, mask, s0, forcings):
    """Raise where holding every zero-mass DOF at 0 would be WRONG, instead of returning it.

    The exponential integrator treats a zero-mass row as a homogeneous Dirichlet row and masks the DOF to
    0. That is exact for ``u = 0`` on the wall and silently wrong for anything else. Measured on a 2-D heat
    block before this guard: with ``u = 1`` on the wall the boundary came back 0, and with ``u = t`` the
    whole field stayed 0 -- plausible numbers, no warning. The same mask would zero any other algebraic
    unknown (a pressure row), whose value is not 0 either.

    Two checks, both on concrete values (under ``jit``/``grad`` the values are tracers and cannot be
    inspected, so a traced call relies on the eager call that precedes it):

    * no zero-mass row may reference an unconstrained DOF -- a Dirichlet row, or the real/imaginary pair of
      a fused complex one, qualifies; a pressure or flux row does not;
    * the data on those rows -- the forcing, sampled at both ends of the window for a time-varying one --
      must be zero. (An initial state that is non-zero on the wall is not refused: it is inconsistent with a
      zero wall value, and every scheme overwrites it at the first step.)
    """
    arrays = [mask, s0, *forcings]
    if any(isinstance(a, jax.core.Tracer) for a in arrays):
        return
    bnd = np.asarray(mask) < 0.5
    if not bnd.any():
        return
    data = getattr(Aop, "data", None)
    idx = getattr(Aop, "indices", None)
    if data is not None and idx is not None and not isinstance(data, jax.core.Tracer):
        rows, cols, vals = np.asarray(idx)[:, 0], np.asarray(idx)[:, 1], np.asarray(data)
        # A zero-mass row may couple to other ZERO-MASS DOFs -- a fused complex Dirichlet row ties the real
        # and imaginary halves of the same DOF -- because with zero data that subsystem is solved by 0.
        # It must not reach an unconstrained DOF: that is an algebraic equation (a pressure row) whose
        # solution is not 0.
        coupled = bnd[rows] & ~bnd[cols] & (np.abs(vals) > 0)
        if coupled.any():
            raise NotImplementedError(
                "jno.solve.exponential: this block has zero-mass rows that couple to other unknowns (an "
                "algebraic constraint -- a pressure or a flux row -- not a plain Dirichlet row). The "
                "exponential integrator holds every zero-mass DOF at 0, which would be wrong for them. Use "
                "jno.solve.theta(...), bdf2() or sdirk(), which impose algebraic rows exactly."
            )
    for v in forcings:
        v = np.asarray(v).reshape(-1)
        scale = max(1.0, float(np.max(np.abs(v)))) if v.size else 1.0
        if v.size and float(np.max(np.abs(v[bnd]))) > 1e-12 * scale:
            raise NotImplementedError(
                "jno.solve.exponential: a Dirichlet value on this block is NOT zero. The exponential "
                "integrator holds every zero-mass DOF at 0, so it would silently return u = 0 there. Use jno.solve.theta(...), bdf2() or sdirk(), or write "
                "the problem for w = u - g with a lifting g of the boundary data so the wall value is 0."
            )


def _stored_runtime_args(block):
    """The block's runtime args at the trainables' STORED values (a parameter's ``.value``, a network's
    module), or ``None`` when one of them has no stored value to read."""
    out = {}
    for name, expr in (getattr(block, "runtime_parameter_exprs", None) or {}).items():
        model = getattr(expr, "model", None)
        module = getattr(model, "module", None)
        if module is None:
            return None
        out[name] = jnp.asarray(module.value) if getattr(model, "_is_parameter", False) else module
    return out


def _refuse_nonsymmetric(A, mask):
    """``symmetric=True`` runs Lanczos, which is exact only for a symmetric free block; refuse any other.

    Measured on two coupled P2/P1 heat fields with a one-way coupling term: the Lanczos result was 1.4e-4
    off with no warning, where ``symmetric=False`` (Arnoldi) is exact. Checked on concrete values only."""
    data, idx = getattr(A, "data", None), getattr(A, "indices", None)
    if data is None or idx is None or any(isinstance(x, jax.core.Tracer) for x in (data, idx, mask)):
        return
    import scipy.sparse as _sp

    free = np.asarray(mask) > 0.5
    i, j, v = np.asarray(idx)[:, 0], np.asarray(idx)[:, 1], np.asarray(data)
    keep = free[i] & free[j]
    S = _sp.coo_matrix((v[keep], (i[keep], j[keep])), shape=A.shape).tocsr()
    gap = abs(S - S.T).max() if S.nnz else 0.0
    if gap > 1e-10 * max(abs(S).max() if S.nnz else 0.0, 1e-300):
        raise NotImplementedError(
            "jno.solve.exponential(symmetric=True): the operator is not symmetric on the free DOFs (relative "
            f"asymmetry {gap / abs(S).max():.1e}: advection, or a one-way coupling between fields), and the "
            "symmetric Lanczos path would return a wrong answer. Use jno.solve.exponential(symmetric=False)."
        )


def _row_abs_sum(M):
    """``sum_j |M_ij|`` per row: zero exactly on a structurally empty (algebraic) row of the mass."""
    data, idx = getattr(M, "data", None), getattr(M, "indices", None)
    if data is not None and idx is not None:
        return jnp.zeros(M.shape[0], jnp.asarray(data).dtype).at[idx[:, 0]].add(jnp.abs(data))
    return jnp.sum(jnp.abs(jnp.asarray(M.todense() if hasattr(M, "todense") else M)), axis=1)


def _refuse_nonpositive_lumping(d, mask):
    """Row-sum lumping needs a positive row sum on every unknown; refuse where the basis does not give one."""
    if isinstance(d, jax.core.Tracer) or isinstance(mask, jax.core.Tracer):
        return
    d, free = np.asarray(d), np.asarray(mask) > 0.5
    bad = free & (d <= 1e-12 * np.max(np.abs(d)))
    if bad.any():
        raise NotImplementedError(
            f"jno.solve.exponential(mass='lumped'): row-sum lumping gives a zero or negative mass on {int(bad.sum())} "
            "of the unknowns -- the row sums of this basis are not positive (a P2 vertex, a Hermite slope or an "
            "edge DOF). Use jno.solve.exponential(mass='consistent'), which integrates the system the other "
            "schemes solve."
        )


def _exponential_integrate(block, args, save_ts, *, order, mass, symmetric, check_only=False):
    """Advance ``M u̇ + A u = f(t)`` per step via ``exp(-dt·M⁻¹A)``: exact for the homogeneous decay, with a
    ``φ₁`` weight for a constant source and a ``φ₂`` ramp weight (ETD2) for a time-varying one.

    * ``lumped`` — symmetric ``L̃ = D^{-1/2} A D^{-1/2}`` (``D`` = row-sum mass); integrate ``w = D^{1/2} u``
      with a fully re-orthogonalized Euclidean Lanczos (:func:`m_inner_funm`). Matrix-free.
    * ``consistent`` — ``L = M⁻¹A`` applied by a **matrix-free M-inner-product Lanczos** (:func:`m_inner_funm`,
      an M-solve per matvec via CG). No lumping error, no factorization.

    Both are **matrix-free and reverse-mode differentiable**; the Dirichlet boundary (zero-mass DOFs) is
    held at 0 by *masking* (a multiply — trace-safe), never a host-side extraction."""
    from .backend_blocks import _block_time_grid
    from .mass import lumped_diagonal, m_inner_funm
    from .matfun import expmv
    from .solver_api import LinearOperator

    if not block.is_linear():
        raise NotImplementedError(
            "jno.solve.exponential integrates a LINEAR block only; use a θ-scheme for a nonlinear one."
        )
    meta = getattr(block, "metadata", None) or {}
    # A runtime-parametric operator A(args) (a trainable coefficient) is fine as long as it does not read
    # time: the assembler says so per term (`operator_reads_time`). A parametric MASS is not (it would have to
    # be re-lumped / re-solved per evaluation, and a trainable mass is refused upstream anyway).
    _param_op = block.operator_fn is not None and meta.get("operator_reads_time") is False
    if (block.operator_fn is not None and not _param_op) or block.mass_fn is not None:
        raise NotImplementedError(
            "jno.solve.exponential needs time-INDEPENDENT M and A (an autonomous system). For time-varying "
            "coefficients use jno.solve.theta(...)."
        )
    if mass not in ("lumped", "consistent"):
        raise ValueError(f"jno.solve.exponential: mass={mass!r} — use 'lumped' or 'consistent'.")

    Mop = block.M
    t_start = float(block.t0)
    Aop = block.operator_fn(t_start, args or {}) if _param_op else block.A
    # The structural guards below read concrete values. Under a training step the args are traced, so they
    # run on the operator at the trainables' STORED values instead -- the structure (symmetry, which rows
    # are constrained) does not depend on the coefficient's value.
    A_check = Aop
    if _param_op and isinstance(getattr(Aop, "data", None), jax.core.Tracer):
        stored = _stored_runtime_args(block)
        A_check = block.operator_fn(t_start, stored) if stored is not None else Aop
    _s0f = getattr(block, "state0_fn", None)
    s0 = jnp.asarray(_s0f(args) if _s0f is not None else block.state0).reshape(-1)
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
    if _time_varying and _param_op and meta.get("forcing_reads_time") is False:
        # the parametric assembly hands every source over as f(t, args); one that reads no time is constant
        c_aff = c_aff + jnp.asarray(block.forcing_vector_fn(t_start, args or {}), dtype).reshape(-1)
        _time_varying = False
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

    d = lumped_diagonal(Mop)
    # Dirichlet DOFs carry NO mass: their whole row of M is empty -- algebraic (u=0), not ODEs. Read off the
    # ABSOLUTE row sum, never the row sum ``d``: the row sum of a higher-order or derivative basis vanishes
    # on unknowns that do carry mass (a P2 vertex, a Hermite slope, an edge DOF), and the mask then held them
    # at 0 -- refused as a "pressure row" on P2, Hermite and Nedelec fields that have no algebraic row at all.
    d_abs = _row_abs_sum(Mop)
    mask = (d_abs > 1e-12 * jnp.max(d_abs)).astype(dtype)  # 1 interior / 0 boundary — a *multiply* (trace-safe)
    if mass == "lumped":
        _refuse_nonpositive_lumping(d, mask)
    if symmetric:
        _refuse_nonsymmetric(A_check, mask)
    # A Krylov space cannot outgrow the problem: cap the size at the number of unknowns (the free ones, when
    # the mask is concrete). Past it the Arnoldi of `expmv` raised outright ("num_matvecs=40 exceeds ... 31").
    n_free = int(np.sum(np.asarray(mask))) if not isinstance(mask, jax.core.Tracer) else n
    order = max(1, min(int(order), n_free))
    _refuse_what_the_mask_would_zero(
        A_check, mask, s0, [c_aff] + ([_f_of(grid[0]), _f_of(grid[-1])] if _time_varying else [])
    )
    if check_only:
        return None

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
            # symmetric L̃ = D^{-1/2} A D^{-1/2}; integrate w = D^{1/2} u with a Euclidean Lanczos
            s = jnp.sqrt(d)
            s_inv = jnp.where(mask > 0, 1.0 / jnp.where(mask > 0, s, 1.0), 0.0)  # 0 on the boundary

            def Ctil_mv(w):
                return s_inv * (Aop @ (s_inv * w))

            e0 = mask / jnp.maximum(jnp.linalg.norm(mask), 1.0)  # no free DOF at all: 0, never 0/0

            def apply_f(vec, fun):
                # f(L̃)·vec by a fully re-orthogonalized Lanczos (the consistent path's, with the Euclidean
                # inner product). A pinned DOF adds an eigenvalue 0 to L̃ and a symmetric mesh repeats
                # eigenvalues, so the subspace can run out before `order` steps; the unrecovered Lanczos of
                # `applyfun` then broke down to NaN (a 3-D P1 heat block), where this one carries on.
                return m_inner_funm(Ctil_mv, lambda a, b: a @ b, e0, vec, fun, order)

            w0, solve_c = s * s0, lambda x: s_inv * x  # forcing in the scaled system: D^{-1/2}·rhs
            to_field = lambda w: s_inv * w  # u = D^{-1/2} w (boundary → 0)
        else:  # consistent — matrix-free M-inner-product Lanczos on L = M⁻¹A: scalable AND differentiable
            m_solve = _consistent_m_solve()
            L_mv = lambda x: m_solve(Aop @ x)  # M⁻¹A x (interior)
            m_inner = lambda a, b: a @ (Mop @ b)  # ⟨a,b⟩_M — M already zeros the boundary
            e0 = mask / jnp.sqrt(jnp.maximum(m_inner(mask, mask), 1e-300))  # a fixed M-unit interior vector (0 if none)
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

    from .backend_blocks import _march_eagerly, _march_to_host

    if _march_eagerly(args, save_ts):
        # Eager: chunks whose frames go to the host as they are made (`_march_to_host`). Each step emits its
        # full field `to_field(w)` -- a fixed diagonal map, so this is the same per-row value as mapping the
        # stacked trajectory afterwards; frames between grid points blend with the shared resampler.
        def make_step(_args, _t_start):
            def step_u(wc, t_target):
                wn, _ = step(wc, t_target)
                return wn, to_field(wn)

            return step_u

        grid_np = np.asarray(grid, dtype=float)
        return _march_to_host(
            block, args, ("exponential", order, mass, symmetric, dt), make_step, w0, grid_np, dtype, save_ts,
            state_of=to_field, prefix_ts=grid_np[:1], prefix_states=[to_field(w0)],
            cache=False,  # the step closes over the source evaluated at THIS call's args
            degree=2,  # ETD2: second order in time
        )  # fmt: skip
    _, ws = jax.lax.scan(step, w0, grid[1:])
    traj_u = jax.vmap(to_field)(jnp.concatenate([w0[None, :], ws], axis=0))  # each row → the full field u
    from .backend_blocks import _resample_trajectory

    return _resample_trajectory(traj_u, grid, save_ts, dtype, 2)  # ETD2: second order in time
