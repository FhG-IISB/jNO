"""Time-integration schemes for the transient path, selected with ``fem.solve(time=...)``.

Default (``time=None``): the θ-method the assembly picks — backward Euler for a first-order system,
trapezoidal for second-order. ``jno.solve.theta(θ)`` overrides θ (``1`` backward Euler, ``1/2``
Crank–Nicolson, ``0`` forward Euler). ``jno.solve.exponential(...)`` advances a **linear autonomous**
parabolic block with the **matrix exponential** — exact in time and unconditionally stable, so it takes
large stiff steps that an implicit θ-step cannot; it reuses :func:`jno.solve.applyfun` (matfree Lanczos).
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
    """Matrix-exponential scheme (see :func:`jno.solve.exponential`) — linear autonomous parabolic only."""

    def __init__(self, order: int, mass: str):
        self.order = int(order)
        self.mass = mass

    def integrate(self, block, args, save_ts, *, linear_solve=None, nonlinear_solve=None):
        return _exponential_integrate(block, args, save_ts, order=self.order, mass=self.mass)

    def __repr__(self):
        return f"jno.solve.exponential(order={self.order}, mass={self.mass!r})"


def _exponential_integrate(block, args, save_ts, *, order, mass):
    """Advance ``M u̇ + A u = c`` exactly per step via ``exp(-dt·M⁻¹A)`` (+ ``φ₁`` forcing).

    * ``lumped`` — symmetric ``L̃ = D^{-1/2} A D^{-1/2}`` (``D`` = row-sum mass); integrate ``w = D^{1/2} u``
      with the Euclidean Lanczos of :func:`applyfun`. Matrix-free.
    * ``consistent`` — ``L = M⁻¹A`` applied by a **matrix-free M-inner-product Lanczos** (:func:`m_inner_funm`,
      an M-solve per matvec via CG). No lumping error, no factorization.

    Both are **matrix-free and reverse-mode differentiable**; the Dirichlet boundary (zero-mass DOFs) is
    held at 0 by *masking* (a multiply — trace-safe), never a host-side extraction."""
    from .backend_blocks import _block_time_grid
    from .mass import lumped_diagonal, m_inner_funm
    from .matfun import applyfun
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

    c = jnp.zeros(n, dtype)  # constant forcing c = affine_bias + forcing(t0)
    if block.affine_bias is not None:
        c = c + jnp.asarray(block.affine_bias, dtype).reshape(-1)
    if block.forcing_vector_fn is not None:
        c = c + jnp.asarray(block.forcing_vector_fn(float(grid[0]), args or {}), dtype).reshape(-1)
    has_forcing = bool(block.affine_bias is not None or block.forcing_vector_fn is not None)

    def _exp(lam):
        return jnp.exp(-dt * lam)

    def _phi1(lam):  # φ₁(z) = (eᶻ − 1)/z at z = −dt·λ (Taylor near 0 for stability)
        z = -dt * lam
        return jnp.where(jnp.abs(z) < 1e-7, 1.0 + 0.5 * z, jnp.expm1(z) / z)

    d = lumped_diagonal(Mop)  # Dirichlet DOFs carry NO mass (d=0) — algebraic (u=0), not ODEs
    mask = (d > 1e-12 * jnp.max(d)).astype(dtype)  # 1 interior / 0 boundary — a *multiply* (trace-safe)

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

        w0, g = s * s0, s_inv * c
        to_field = lambda w: s_inv * w  # u = D^{-1/2} w (boundary → 0)
    else:  # consistent — matrix-free M-inner-product Lanczos on L = M⁻¹A: scalable AND differentiable
        from jax.scipy.sparse.linalg import cg as _cg

        from .linear import matrix_diagonal

        Mreg = lambda x: Mop @ x + (1.0 - mask) * x  # M + (1-mask)·I — SPD (boundary block is identity)
        jac = 1.0 / (matrix_diagonal(Mop) + (1.0 - mask))  # Jacobi preconditioner for the CG M-solve

        def m_solve(rhs):  # M⁻¹·rhs on the interior (boundary → 0, since the masked rhs is 0 there)
            return _cg(Mreg, rhs, tol=1e-10, maxiter=300, M=lambda z: jac * z)[0]

        L_mv = lambda x: m_solve(mask * (Aop @ x))  # M⁻¹A x (interior)
        m_inner = lambda a, b: a @ (Mop @ b)  # ⟨a,b⟩_M — M already zeros the boundary
        e0 = mask / jnp.sqrt(m_inner(mask, mask))  # a fixed M-unit interior vector
        apply_f = lambda vec, fun: m_inner_funm(L_mv, m_inner, e0, vec, fun, order)
        w0, g = mask * s0, m_solve(mask * c)  # integrate u directly (homogeneous boundary); forcing M⁻¹c
        to_field = lambda u: u

    def step(w, _t):
        wn = apply_f(w, _exp)
        if has_forcing:
            wn = wn + dt * apply_f(g, _phi1)
        return wn, wn

    _, ws = jax.lax.scan(step, w0, grid[1:])
    traj_u = jax.vmap(to_field)(jnp.concatenate([w0[None, :], ws], axis=0))  # each row → the full field u
    save_ts = jnp.asarray(save_ts, dtype)
    return jax.vmap(lambda col: jnp.interp(save_ts, grid, col), in_axes=1, out_axes=1)(traj_u)
