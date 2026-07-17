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

    Lumped mass ``D`` (row sums of ``M``) makes ``L̃ = D^{-1/2} A D^{-1/2}`` symmetric, so we integrate the
    ODE ``ẇ = -L̃ w + g`` in the variable ``w = D^{1/2} u`` (``g = D^{-1/2} c``) with :func:`applyfun`'s
    symmetric Lanczos, then map back ``u = D^{-1/2} w``. Reverse-mode differentiable."""
    from .backend_blocks import _block_time_grid
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
    if mass != "lumped":
        raise NotImplementedError("jno.solve.exponential supports mass='lumped' (V1); consistent-mass is planned.")

    Mop, Aop = block.M, block.A
    s0 = jnp.asarray(block.state0).reshape(-1)
    dtype = s0.dtype
    n = s0.shape[0]
    dt = float(block.dt)
    grid = jnp.asarray(_block_time_grid(block), dtype)

    d = Mop @ jnp.ones(n, dtype)  # row-sum (lumped) mass diagonal
    # Dirichlet DOFs carry NO mass (d=0) — they are algebraic (u=0), not ODEs. Zeroing s_inv there
    # restricts the exponential to the interior and holds the (homogeneous) boundary at 0.
    interior = d > 1e-12 * jnp.max(d)
    s = jnp.sqrt(d)
    s_inv = jnp.where(interior, 1.0 / jnp.where(interior, s, 1.0), 0.0)
    Ltilde = LinearOperator.from_matvec(lambda w: s_inv * (Aop @ (s_inv * w)), shape=(n, n))  # symmetric

    c = jnp.zeros(n, dtype)  # constant forcing c = affine_bias + forcing(t0)
    if block.affine_bias is not None:
        c = c + jnp.asarray(block.affine_bias, dtype).reshape(-1)
    if block.forcing_vector_fn is not None:
        c = c + jnp.asarray(block.forcing_vector_fn(float(grid[0]), args or {}), dtype).reshape(-1)
    has_forcing = bool(block.affine_bias is not None or block.forcing_vector_fn is not None)
    g = s_inv * c

    def _exp(lam):
        return jnp.exp(-dt * lam)

    def _phi1(lam):  # φ₁(z) = (eᶻ − 1)/z at z = −dt·λ (Taylor near 0 for stability)
        z = -dt * lam
        return jnp.where(jnp.abs(z) < 1e-7, 1.0 + 0.5 * z, jnp.expm1(z) / z)

    e0 = interior.astype(dtype)
    e0 = e0 / jnp.linalg.norm(e0)  # a fixed nonzero interior unit vector (fallback for a null start)

    def _apply(vec, fun):
        # f(L̃)·vec by Lanczos, but applied to the NORMALISED vector and scaled back — Lanczos divides by
        # the start-vector norm, so a zero/decayed vec would give NaN; this is exact (linearity) and 0 at 0.
        nrm = jnp.linalg.norm(vec)
        unit = jnp.where(nrm > 1e-300, vec / jnp.where(nrm > 1e-300, nrm, 1.0), e0)
        return nrm * applyfun(Ltilde, unit, fun=fun, order=order)

    def step(w, _t):
        wn = _apply(w, _exp)
        if has_forcing:
            wn = wn + dt * _apply(g, _phi1)
        return wn, wn

    _, ws = jax.lax.scan(step, s * s0, grid[1:])  # scan in w = D^{1/2} u
    traj_u = jnp.concatenate([(s * s0)[None, :], ws], axis=0) * s_inv[None, :]  # back to u = D^{-1/2} w
    save_ts = jnp.asarray(save_ts, dtype)
    return jax.vmap(lambda col: jnp.interp(save_ts, grid, col), in_axes=1, out_axes=1)(traj_u)
