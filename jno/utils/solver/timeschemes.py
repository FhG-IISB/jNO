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

    Both mass treatments reduce the pencil to a **symmetric** operator ``C`` in a transformed variable
    ``w``, integrate ``ẇ = -C w + g`` with :func:`applyfun`'s Lanczos, and map ``w`` back to the field
    ``u`` — ``lumped`` uses ``C = D^{-1/2} A D^{-1/2}`` (``w = D^{1/2} u``, matrix-free), ``consistent``
    factors ``M = L Lᵀ`` once and uses ``C = L⁻¹ A L⁻ᵀ`` (``w = Lᵀ u``). Reverse-mode differentiable
    (``lumped``); ``consistent`` needs a concrete operator (host Cholesky)."""
    import numpy as np

    from .backend_blocks import _block_time_grid
    from .mass import _dense, cholesky_spd, lumped_diagonal
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

    d = lumped_diagonal(Mop)  # Dirichlet DOFs carry NO mass (d=0) — algebraic (u=0), not ODEs.
    if mass == "lumped":
        interior = d > 1e-12 * jnp.max(d)
        s = jnp.sqrt(d)
        s_inv = jnp.where(interior, 1.0 / jnp.where(interior, s, 1.0), 0.0)  # 0 on the boundary
        C = LinearOperator.from_matvec(lambda w: s_inv * (Aop @ (s_inv * w)), shape=(n, n))
        w0, g = s * s0, s_inv * c
        to_field = lambda w: s_inv * w  # u = D^{-1/2} w (boundary → 0)
        e0 = interior.astype(dtype)
    else:  # consistent: restrict to the interior (zero-mass Dirichlet DOFs), Cholesky M_ii = L Lᵀ (once)
        from jax.scipy.linalg import solve_triangular

        try:
            interior_np = np.asarray(d) > 1e-12 * float(np.asarray(d).max())
        except Exception as e:
            raise RuntimeError(
                "jno.solve.exponential(mass='consistent') factors M on the host from a concrete operator; it "
                "cannot run under a trace (parametric solve). Use mass='lumped' there."
            ) from e
        idx = np.where(interior_np)[0]
        idxj = jnp.asarray(idx)
        Md, Ad = _dense(Mop), _dense(Aop)
        L = cholesky_spd(Md[idxj][:, idxj])
        A_ii = Ad[idxj][:, idxj]
        C = LinearOperator.from_matvec(
            lambda w: solve_triangular(L, A_ii @ solve_triangular(L.T, w, lower=False), lower=True),
            shape=(len(idx), len(idx)),
        )
        w0 = L.T @ s0[idxj]
        g = solve_triangular(L, c[idxj], lower=True)  # L⁻¹ c_i
        to_field = lambda w: jnp.zeros(n, dtype).at[idxj].set(solve_triangular(L.T, w, lower=False))  # boundary → 0
        e0 = jnp.ones(len(idx), dtype)

    e0 = e0 / jnp.linalg.norm(e0)  # fixed nonzero unit vector — fallback for a null/decayed Lanczos start

    def _exp(lam):
        return jnp.exp(-dt * lam)

    def _phi1(lam):  # φ₁(z) = (eᶻ − 1)/z at z = −dt·λ (Taylor near 0 for stability)
        z = -dt * lam
        return jnp.where(jnp.abs(z) < 1e-7, 1.0 + 0.5 * z, jnp.expm1(z) / z)

    def _apply(vec, fun):
        # f(C)·vec by Lanczos, applied to the NORMALISED vector and scaled back — Lanczos divides by the
        # start-vector norm, so a zero/decayed vec would give NaN; this is exact (linearity) and 0 at 0.
        nrm = jnp.linalg.norm(vec)
        unit = jnp.where(nrm > 1e-300, vec / jnp.where(nrm > 1e-300, nrm, 1.0), e0)
        return nrm * applyfun(C, unit, fun=fun, order=order)

    def step(w, _t):
        wn = _apply(w, _exp)
        if has_forcing:
            wn = wn + dt * _apply(g, _phi1)
        return wn, wn

    _, ws = jax.lax.scan(step, w0, grid[1:])
    traj_u = jax.vmap(to_field)(jnp.concatenate([w0[None, :], ws], axis=0))  # each w-row → the full field u
    save_ts = jnp.asarray(save_ts, dtype)
    return jax.vmap(lambda col: jnp.interp(save_ts, grid, col), in_axes=1, out_axes=1)(traj_u)
