"""Finite-difference PDE solver — the strong-form sibling of :func:`jno.fem`.

Write the **strong-form residual** ``R(u)`` (``u`` is the nodal field) with the FD operators in this
module; :func:`fdm` folds in Dirichlet BCs and hands the residual straight to the ``jno.solve``
**nonlinear driver** — the *same* Newton–Krylov + implicit-``custom_root`` machinery :func:`jno.fem`
uses, so **linear and nonlinear** problems are handled uniformly (a linear residual converges in one
Newton step). Collocation at the mesh nodes means no quadrature, no test functions, no mass matrix
(so it is leaner than the weak-form assembler).

Because the residual is a plain differentiable function of the DOFs and the solve differentiates
through ``custom_root``, gradients to parameters inside it (a source, a coefficient field,
``jno.nn.wrap`` net) flow through — so ``jno.fdm`` composes into ``jno.core`` for inverse problems,
exactly like ``jno.fem.solve()``.

v1 scope: scalar, steady, 2-D triangular mesh, Dirichlet BCs (linear + nonlinear residuals).
Neumann/Robin, periodic, transient (method-of-lines, ``M = I``), and a structured-grid stencil backend
are planned extensions (see ``plans/fdm-solver.md``).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from . import solve as _solve
from .differential_operators import DifferentialOperators as _D

__all__ = ["fdm", "laplacian", "gradient"]


def _mesh(domain):
    mc = domain.mesh_connectivity
    dim = int(getattr(domain, "dimension", 2))
    pts = jnp.asarray(np.asarray(mc["points"])[:, :dim])
    if dim != 2:
        raise NotImplementedError("jno.fdm: only 2-D triangular meshes are supported in v1.")
    return pts, jnp.asarray(mc["triangles"])


def laplacian(u, domain, method: str = "cotangent"):
    """FD Laplacian ``Δu`` of the nodal field ``u`` on the domain's mesh. ``method="cotangent"``
    (symmetric, accurate — CG-compatible) or ``"gradient_of_gradient"`` / ``"lsq_of_gradient"``."""
    pts, tris = _mesh(domain)
    return _D.compute_fd_laplacian_2d_simple(u, pts, tris, dims=(0, 1), method=method)


def gradient(u, domain):
    """FD gradient ``∇u`` of the nodal field ``u`` — shape ``(N, 2)``."""
    pts, tris = _mesh(domain)
    return _D.compute_fd_gradient_2d_simple(u, pts, tris, dims=(0, 1))


class FDMSystem:
    """A finite-difference discretization built from a strong-form residual + Dirichlet BCs.

    ``residual(u_dofs) -> R_at_nodes`` must be a differentiable function of the nodal DOF vector, zero
    at the PDE solution (e.g. ``lambda u: -jno.fdm.laplacian(u, d) - f``). ``dirichlet`` maps a
    boundary-region tag to a value (scalar, or a coordinate function ``g(x, y)``). ``initial`` is the
    time-dependent IC (a nodal array or a coordinate function ``u0(x, y)``); with it, a domain that
    carries a ``time=(t0, t1, n)`` axis makes ``.solve()`` march in time (method of lines)."""

    def __init__(self, domain, residual, dirichlet=None, initial=None):
        self.domain = domain
        self.residual = residual
        self.dirichlet = dict(dirichlet or {})
        self.initial = initial
        self._N = int(np.asarray(domain.mesh_connectivity["points"]).shape[0])

    def _initial_state(self):
        if self.initial is None:
            raise ValueError(
                "jno.fdm: a time-dependent domain (time=...) needs an initial condition — "
                "pass initial=<nodal array or u0(x, y[, z])>."
            )
        if callable(self.initial):
            pts = np.asarray(self.points)
            return jnp.asarray(self.initial(*[pts[:, k] for k in range(pts.shape[1])]))
        return jnp.asarray(self.initial).reshape(-1)

    @property
    def points(self):
        """DOF (mesh-node) coordinates, shape ``(N, dim)``."""
        pts, _ = _mesh(self.domain)
        return pts

    def _region_nodes(self, tag):
        reg = getattr(self.domain, "_boundary_registry", {}).get(tag)
        if reg is not None and "point_indices" in reg and len(reg["point_indices"]) > 0:
            return np.asarray(reg["point_indices"], dtype=int)
        return np.asarray(self.domain.mesh_connectivity["boundary_indices"], dtype=int)  # whole boundary

    def _dirichlet_mask_values(self):
        pts = np.asarray(self.points)
        mask = np.zeros(self._N, dtype=bool)
        vals = np.zeros(self._N)
        for tag, g in self.dirichlet.items():
            idx = self._region_nodes(tag)
            mask[idx] = True
            vals[idx] = g(*[pts[idx, k] for k in range(pts.shape[1])]) if callable(g) else float(g)
        return jnp.asarray(mask), jnp.asarray(vals)

    def solve(self, nonlinear=None, x0=None, *, save_ts=None):
        """Single entry — **steady or transient, inferred from the domain** (like ``jno.fem.solve()``).

        If the domain carries a ``time=(t0, t1, n)`` axis, ``.solve()`` marches in time (method of
        lines): ``t_span``/``dt`` come from ``domain.time`` (via ``_infer_time_window``) and the IC from
        ``initial`` — no explicit time args. Otherwise it solves the steady system.

        Both paths reuse the SAME ``jno.solve`` Newton–Krylov + implicit-``custom_root`` machinery
        ``jno.fem`` uses (linear and nonlinear handled uniformly; a linear residual converges in one
        step; fully matrix-free). Dirichlet is folded into the residual (boundary rows become
        ``u - g``); the transient path additionally reuses ``jno.fem``'s ``SemidiscreteTimeBlock``
        stepper. Gradients to parameters in ``residual`` flow through ``custom_root``, so ``jno.fdm``
        composes into ``jno.core`` for (time-dependent) inverse problems.

        Args:
            nonlinear: nonlinear driver slot (default :func:`jno.solve.newton`) — steady path only.
            x0: initial guess for the steady solve (default zeros).
            save_ts: transient sample times (default: the ``domain.time`` grid).
        """
        from .utils.solver.time_route import _infer_time_window

        if getattr(self.domain, "time", None) is not None:
            t0, t1, dt = _infer_time_window(self.domain)
            if dt is None:
                raise ValueError("jno.fdm transient: domain.time must have n_points >= 2.")
            return self._march(self._initial_state(), t0, t1, dt, save_ts)

        bmask, bvals = self._dirichlet_mask_values()

        def residual_with_bc(u):
            return jnp.where(bmask, u - bvals, self.residual(u))  # Dirichlet rows: u - g

        driver = nonlinear or _solve.newton()
        u0 = jnp.zeros(self._N) if x0 is None else jnp.asarray(x0)
        return driver(residual_with_bc, u0)

    def solve_transient(self, u0, t_span, nsteps, *, save_ts=None):
        """Explicit method-of-lines march (when you don't drive time through ``domain.time``). See
        :meth:`solve`; ``dt = (t1 - t0) / nsteps``."""
        t0, t1 = float(t_span[0]), float(t_span[1])
        return self._march(jnp.asarray(u0), t0, t1, (t1 - t0) / int(nsteps), save_ts)

    def _march(self, u0, t0, t1, dt, save_ts):
        """Method-of-lines transient of ``u̇ = 𝒩(u) = -residual(u)`` (the SAME residual serves steady
        and transient). Reuses jNO's solver-agnostic :class:`SemidiscreteTimeBlock` + backward-Euler
        ``lax.scan`` integrator — no new time-stepping code, and ``custom_root`` differentiable.
        Collocation gives ``M = I`` (leaner than FEM); Dirichlet nodes carry a zero mass row so they
        stay pinned at ``g``. Autonomous residuals only in v1 (no explicit ``t`` / time-varying source)."""
        import jax.experimental.sparse as jsparse

        from .utils.solver.backend_blocks import (
            SemidiscreteTimeBlock,
            _block_time_grid,
            _default_transient_integrate,
        )

        bmask, bvals = self._dirichlet_mask_values()
        diag = jnp.stack([jnp.arange(self._N), jnp.arange(self._N)], axis=1)
        M = jsparse.BCOO((jnp.where(bmask, 0.0, 1.0), diag), shape=(self._N, self._N))  # 0 mass on Dirichlet rows

        def residual(wn, t, args):  # M u̇ + residual = 0  →  interior u̇ = -R = 𝒩;  boundary wn = g
            return jnp.where(bmask, wn - bvals, self.residual(wn))

        block = SemidiscreteTimeBlock(
            mass=lambda t, args: M, residual=residual, state0=jnp.asarray(u0), t0=float(t0), t1=float(t1), dt=float(dt)
        )
        ts = _block_time_grid(block) if save_ts is None else jnp.asarray(save_ts)
        return _default_transient_integrate(block, {}, ts)


def fdm(domain, residual, dirichlet=None, initial=None) -> FDMSystem:
    """Build a finite-difference system from a strong-form residual + Dirichlet BCs — the strong-form
    sibling of :func:`jno.fem`. Pass ``initial=`` for a transient problem on a ``time=``-carrying
    domain (``.solve()`` then marches in time). See :class:`FDMSystem`."""
    return FDMSystem(domain, residual, dirichlet, initial)


fdm.laplacian = laplacian  # convenience: jno.fdm.laplacian(u, domain)
fdm.gradient = gradient
