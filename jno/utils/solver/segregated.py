"""Segregated pressure-correction solvers for saddle systems: ``jno.solve.simple`` and ``jno.solve.piso``.

Based on:
  Patankar, S. V., & Spalding, D. B. (1972). A calculation procedure for heat, mass and momentum transfer in
  three-dimensional parabolic flows. International Journal of Heat and Mass Transfer 15(10), 1787-1806.
  Issa, R. I. (1986). Solution of the implicitly discretised fluid flow equations by operator-splitting.
  Journal of Computational Physics 62(1), 40-65.

Both work on the assembled tangent of any saddle system ``R(u, p) = 0``, split by its constraint rows into
``J = [[A, B], [C, D]]`` (``u`` the primary unknowns, ``p`` the multipliers): a momentum predictor with ``p``
held, then pressure corrections on the Schur complement ``S = D − C diag(A)⁻¹ B``, the ``diag(A)``
approximation of SIMPLE. Nothing in them names a velocity or a pressure; the problem says which rows are
constraints (``jno.fvm`` finds them by structure).
"""

from __future__ import annotations

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np


def _plan(indices, cons):
    """Host-side structure of the split ``[[A, B], [C, D]]`` of a Jacobian with the (static) sparsity pattern
    ``indices`` ``(nnz, 2)``: each block's entry positions and compact indices, ``A``'s diagonal, and the
    pattern of the Schur complement ``S = D - C diag(A)^-1 B`` with the product triples that fill it."""
    idx = np.asarray(indices)
    prim = ~cons
    rmap = {True: np.cumsum(cons) - 1, False: np.cumsum(prim) - 1}
    r_is, c_is = cons[idx[:, 0]], cons[idx[:, 1]]
    blocks = {}
    for name, (rc, cc) in {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}.items():
        pos = np.flatnonzero((r_is == rc) & (c_is == cc))
        blocks[name] = (pos, np.stack([rmap[rc][idx[pos, 0]], rmap[cc][idx[pos, 1]]], axis=1))
    nu, npr = int(prim.sum()), int(cons.sum())
    Apos, Aidx = blocks["A"]
    diag = Apos[Aidx[:, 0] == Aidx[:, 1]], Aidx[Aidx[:, 0] == Aidx[:, 1], 0]
    # S = D - C diag(A)^-1 B: entries (i, j) from every C_ik B_kj
    Cpos, Cidx = blocks["C"]
    Bpos, Bidx = blocks["B"]
    by_k = {}
    for e, (k, jj) in enumerate(Bidx):
        by_k.setdefault(int(k), []).append((e, int(jj)))
    trip = [(ci, int(i), int(k), be, jj) for ci, (i, k) in enumerate(Cidx) for be, jj in by_k.get(int(k), [])]
    Dpos, Didx = blocks["D"]
    keys = {tuple(r): n for n, r in enumerate(Didx.tolist())}
    for _, i, _, _, jj in trip:
        keys.setdefault((i, jj), len(keys))
    Sidx = np.array(sorted(keys, key=keys.get), dtype=np.int64).reshape(-1, 2)
    t = np.array([(keys[(i, jj)], ci, k, be) for ci, i, k, be, jj in trip], dtype=np.int64).reshape(-1, 4)
    return dict(blocks=blocks, diag=diag, nu=nu, np=npr, Sidx=Sidx, Dslot=np.arange(len(Dpos)), trip=t)


def _bcoo(data, idx, shape):
    return jsparse.BCOO((data, jnp.asarray(idx)), shape=shape)


class _Segregated:
    """A configured SIMPLE / PISO driver (see :func:`jno.solve.simple`, :func:`jno.solve.piso`)."""

    def __init__(self, name, *, alpha_u, alpha_p, corrections, max_iters, rtol, atol, linear=None, constraint=None):
        self.name, self.alpha_u, self.alpha_p = name, float(alpha_u), float(alpha_p)
        self.corrections, self.max_iters, self.rtol, self.atol = int(corrections), int(max_iters), float(rtol), float(atol)
        self.linear, self.constraint = linear, constraint
        self.direct = True  # it reads the assembled tangent
        self.config = dict(alpha_u=alpha_u, alpha_p=alpha_p, corrections=corrections, max_iters=max_iters)

    def with_constraint(self, rows, indices):
        """This driver for a system whose constraint (multiplier) rows are the boolean mask ``rows``, with the
        static sparsity pattern ``indices`` ``(nnz, 2)`` of its assembled tangent."""
        out = _Segregated(
            self.name, alpha_u=self.alpha_u, alpha_p=self.alpha_p, corrections=self.corrections,
            max_iters=self.max_iters, rtol=self.rtol, atol=self.atol, linear=self.linear,
            constraint=np.asarray(rows, dtype=bool),
        )  # fmt: skip
        out.plan = _plan(indices, out.constraint)
        return out

    def _solve(self, A, b):
        from ...solve import lu

        lin = self.linear if self.linear is not None else lu()
        return lin(A, b)

    def _sweep(self, residual_fn, jacobian, w):
        """One predictor and ``corrections`` pressure corrections from the state ``w``."""
        P = self.plan
        nu, npr = P["nu"], P["np"]
        iu = jnp.asarray(np.flatnonzero(~self.constraint))
        ip = jnp.asarray(np.flatnonzero(self.constraint))
        data = jacobian(w).data
        (Apos, Aidx), (Bpos, Bidx), (Cpos, Cidx), (Dpos, Didx) = (P["blocks"][k] for k in "ABCD")
        Ad = data[Apos]
        dA = jnp.zeros(nu, data.dtype).at[jnp.asarray(P["diag"][1])].add(data[P["diag"][0]])
        if self.alpha_u != 1.0:  # implicit under-relaxation of the predictor
            on = Aidx[:, 0] == Aidx[:, 1]
            Ad = Ad + jnp.where(jnp.asarray(on), dA[jnp.asarray(Aidx[:, 0])] * (1.0 / self.alpha_u - 1.0), 0.0)
            dA = dA / self.alpha_u
        inv = 1.0 / jnp.where(dA != 0, dA, 1.0)
        A = _bcoo(Ad, Aidx, (nu, nu))
        B = _bcoo(data[Bpos], Bidx, (nu, npr))
        t = P["trip"]
        Sdata = jnp.zeros(P["Sidx"].shape[0], data.dtype).at[jnp.asarray(P["Dslot"])].add(data[Dpos])
        if t.size:
            prod = data[Cpos][t[:, 1]] * inv[t[:, 2]] * data[Bpos][t[:, 3]]
            Sdata = Sdata.at[jnp.asarray(t[:, 0])].add(-prod)
        S = _bcoo(Sdata, P["Sidx"], (npr, npr))
        r = residual_fn(w)
        w = w.at[iu].add(self._solve(A, -r[iu]))
        for _ in range(self.corrections):
            dp = self._solve(S, -residual_fn(w)[ip])
            w = w.at[iu].add(-inv * (B @ dp))
            w = w.at[ip].add(self.alpha_p * dp)
        return w

    def __call__(self, residual_fn, u0, *, linear_solve=None, jacobian=None, **_):
        if jacobian is None or getattr(self, "plan", None) is None:
            raise ValueError(
                f"jno.solve.{self.name}: needs the assembled tangent and the system's constraint rows; it composes "
                "with jno.fvm, which supplies both."
            )
        w = jnp.asarray(u0)
        if self.name == "piso":
            # A time step: sweeps of one predictor and n_corr corrections, a fixed number of them, each kept only
            # while the step is not yet converged, so the loop stays traceable and differentiable.
            r0 = jnp.linalg.norm(residual_fn(w))

            def body(_, w):
                done = jnp.linalg.norm(residual_fn(w)) <= self.atol + self.rtol * r0
                return jnp.where(done, w, self._sweep(residual_fn, jacobian, w))

            return jax.lax.fori_loop(0, self.max_iters, body, w)
        if isinstance(w, jax.core.Tracer):
            raise NotImplementedError(
                f"jno.solve.{self.name}: the outer iteration runs eagerly (it decides convergence on values); under a "
                "trace (a gradient through the solve) use the default Newton, jno.solve.newton(direct=True)."
            )
        r0 = float(jnp.linalg.norm(residual_fn(w)))
        sweep = jax.jit(lambda z: self._sweep(residual_fn, jacobian, z))  # one compile, then every sweep reuses it
        for it in range(self.max_iters):
            w = sweep(w)
            rn = float(jnp.linalg.norm(residual_fn(w)))
            if not np.isfinite(rn):
                raise RuntimeError(
                    f"jno.solve.{self.name}: the iteration diverged at sweep {it + 1}; lower alpha_u / alpha_p."
                )
            if rn <= self.atol + self.rtol * r0:
                return w
        raise RuntimeError(
            f"jno.solve.{self.name} did not converge in max_iters={self.max_iters}: residual {rn:.3e} against "
            f"{self.atol + self.rtol * r0:.3e}. Raise max_iters, or change alpha_u / alpha_p."
        )

    def __repr__(self):
        return f"jno.solve.{self.name}({', '.join(f'{k}={v!r}' for k, v in self.config.items())})"
