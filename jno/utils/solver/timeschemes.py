"""Time-integration schemes for the transient path, selected with ``fem.solve(time=...)``.

Default (``time=None``): the θ-method the assembly picks — backward Euler for a first-order system,
trapezoidal for second-order. ``jno.solve.theta(θ)`` overrides θ (``1`` backward Euler, ``1/2``
Crank–Nicolson, ``0`` forward Euler). ``jno.solve.exponential(...)`` advances a linear parabolic block
with **time-independent** ``M``, ``A`` by the **matrix exponential** — its homogeneous decay exact in time
and unconditionally stable, so it takes large stiff steps that an implicit θ-step cannot (a time-varying
source is integrated by ETD2). For a symmetric operator it reuses the Lanczos
:func:`jno.solve.applyfun`; for a **non-symmetric** one (``symmetric=False``, advection–diffusion) it uses
an Arnoldi + differentiable **Padé** exponential (:func:`jno.utils.solver.matfun.expmv`), all matfree.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class _ThetaScheme:
    """θ-method scheme (see :func:`jno.solve.theta`) — reuses the default stepper with an explicit θ."""

    def __init__(self, theta: float):
        self.theta = float(theta)

    def integrate(self, block, args, save_ts, *, linear_solve, nonlinear_solve):
        from .backend_blocks import _default_transient_integrate

        return _default_transient_integrate(
            block, args, save_ts, linear_solve=linear_solve, nonlinear_solve=nonlinear_solve, theta=self.theta
        )

    def __repr__(self):
        return f"jno.solve.theta({self.theta})"


class _ExponentialScheme:
    """Matrix-exponential scheme (see :func:`jno.solve.exponential`) — linear, time-independent ``M``/``A``
    (a time-varying source is handled by ETD2)."""

    def __init__(self, order: int, mass: str, symmetric: bool):
        self.order = int(order)
        self.mass = mass
        self.symmetric = bool(symmetric)

    def integrate(self, block, args, save_ts, *, linear_solve=None, nonlinear_solve=None):
        return _exponential_integrate(block, args, save_ts, order=self.order, mass=self.mass, symmetric=self.symmetric)

    def __repr__(self):
        return f"jno.solve.exponential(order={self.order}, mass={self.mass!r}, symmetric={self.symmetric})"


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
