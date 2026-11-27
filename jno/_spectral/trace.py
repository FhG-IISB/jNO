"""``jno.spectral([...])``: spectral collocation of a strong-form constraint list.

The constraint list is written exactly as for :func:`jno.fdm`; the unknowns are spectral
(``domain.unknown(space="spectral", order=...)``). The problem

  * fixes every unknown's basis from the written periodic ties (a tied axis is Fourier, any other Chebyshev,
    unless named) -- the tie itself then holds by construction and is dropped;
  * evaluates every constraint on the full tensor grid of the node set its coordinates name, with the
    unknowns and their derivatives taken from the per-axis basis operators (:class:`_SpectralEvaluator`), and
    keeps the rows of its region: the PDE on every node of its unknown, boundary rows replacing the PDE rows
    of their nodes (Dirichlet winning at corners, as in ``jno.fdm``), a gauge (``psi.mean``) replacing one;
  * solves the result matrix-free (JVP + Krylov) or, small, by a dense factorisation.

Based on:
  Trefethen, L. N. (2000). Spectral Methods in MATLAB. SIAM -- collocation with boundary rows (ch. 7, 13).
  Boyd, J. P. (2001). Chebyshev and Fourier Spectral Methods (2nd ed.). Dover -- boundary bordering and the
  3/2 rule (ch. 6, 11).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .._strong_form import (
    StrongFormProblem,
    _find_unknowns,
    _iter,
    _region_tag,
    _spatial_tags,
    _unwrap,
    _zero_temporal,
)
from ..trace_evaluator import TraceEvaluator
from . import fourier_nd as fnd
from .domain import FACE_NAMES, region_points
from .scheme import FAMILY

__all__ = ["spectral"]

_DENSE_MAX = 4096  # DOFs up to which the default linear solve assembles and factorises (measured: P2 tests)


def _is_spectral_model(model) -> bool:
    return "_spectral_space" in getattr(model, "__dict__", {})


def _models_in(node):
    from ..trace import ModelCall

    out, stack, seen = [], [_unwrap(node)], set()
    while stack:
        n = _unwrap(stack.pop())
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, ModelCall) and _is_spectral_model(n.model):
            if all(m is not n.model for m in out):
                out.append(n.model)
        stack.extend(_iter(n))
    return out


def _is_global(c) -> bool:
    """A constraint whose value is one number (a reduction such as ``psi.mean`` or an integral): a gauge row."""
    from ..trace import FunctionCall, Integral

    stack, seen = [_unwrap(c)], set()
    while stack:
        n = _unwrap(stack.pop())
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, Integral) or (isinstance(n, FunctionCall) and getattr(n, "_name", None) in ("mean", "sum")):
            return True
        stack.extend(_iter(n))
    return False


def _dealias_of(node):
    """The largest ``jno.sp(dealias=a)`` on a derivative of the expression (None if none)."""
    from ..trace import Hessian, Jacobian

    best, stack, seen = None, [_unwrap(node)], set()
    while stack:
        n = _unwrap(stack.pop())
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, (Jacobian, Hessian)):
            a = getattr(n.scheme, "dealias", None)
            if a is not None:
                best = a if best is None else max(best, a)
        stack.extend(_iter(n))
    return best


def _var_axis(var):
    """The coordinate axis a spatial Variable differentiates along (``None`` for time or a normal)."""
    if getattr(var, "axis", "spatial") == "temporal" or str(getattr(var, "tag", "")).startswith("n_"):
        return None
    return int(var.dim[0])


def derivative_chain(node, ndim):
    """``(model, alpha, component)`` when ``node`` is a pure derivative chain of a spectral unknown (``u``, ``u.x``,
    ``u.xy``, ``(u.x).x``, ``U[..., 1].y``), else ``None``: such a node is one leaf, taken on the unknown's own
    basis. ``component`` is ``None`` for the whole field."""
    from ..trace import FunctionCall, Hessian, Jacobian, ModelCall

    n = _unwrap(node)
    if isinstance(n, ModelCall) and _is_spectral_model(n.model):
        return n.model, (0,) * ndim, None
    if isinstance(n, FunctionCall) and getattr(n, "_name", None) == "getitem":
        key = getattr(n, "getitem_key", None)
        if isinstance(key, tuple) and len(key) == 2 and key[0] is Ellipsis and isinstance(key[1], int):
            inner = derivative_chain(n.args[0], ndim)
            if inner is not None and inner[2] is None:
                return inner[0], inner[1], key[1]
        return None
    if isinstance(n, Jacobian) and len(n.variables) == 1:
        ax = _var_axis(n.variables[0])
        inner = derivative_chain(n.target, ndim) if ax is not None else None
        if inner is not None:
            a = list(inner[1])
            a[ax] += 1
            return inner[0], tuple(a), inner[2]
    if isinstance(n, Hessian) and (len(n.variables) == 1 or (len(n.variables) == 2 and not n.trace)):
        axes = [_var_axis(v) for v in n.variables]
        inner = derivative_chain(n.target, ndim) if None not in axes else None
        if inner is not None:
            a = list(inner[1])
            for ax in axes * (2 if len(axes) == 1 else 1):
                a[ax] += 1
            return inner[0], tuple(a), inner[2]
    return None


class _SpectralEvaluator(TraceEvaluator):
    """:class:`TraceEvaluator` on one tensor grid: a spectral unknown evaluates to its interpolant on the grid,
    its derivatives come from its own basis (then move to the grid), and a derivative of any other expression
    of the grid is taken with the grid's own per-axis operators. Leaves are computed once per evaluator."""

    def __init__(self, params, problem, grid, seeded=None):
        super().__init__(params)
        self._problem, self._grid = problem, grid
        self._leaves = {}
        # coefficients (fourier_nd layout) an all-Fourier march already holds: no forward transform for them
        for lid, ch in (seeded or {}).items():
            self._leaves[("coeffs", lid)] = ch

    # values -------------------------------------------------------------------------------------------------
    def _eval_flax_module_call(self, expr, ctx):
        if _is_spectral_model(expr.model):
            return self._leaf(expr.model, (0,) * self._grid.ndim)
        return super()._eval_flax_module_call(expr, ctx)

    def _eval_function_call(self, expr, ctx):
        chain = self._chain(expr) if getattr(expr, "_name", None) == "getitem" else None
        if chain is not None:  # a component of a spectral unknown (or of its derivative): one leaf, sliced
            model, alpha, comp = chain
            return self._leaf(model, alpha)[..., comp : comp + 1 if comp != -1 else None]
        return super()._eval_function_call(expr, ctx)

    def _leaf(self, model, alpha):
        key = (model.layer_id, tuple(alpha))
        hit = self._leaves.get(key)
        if hit is not None:
            return hit
        own = self._problem._grid_of(model)
        module = self.params.get(model.layer_id, model.module)
        vals = jnp.asarray(module.value)
        vshape = vals.shape[1:]
        v = vals.reshape(own.size, -1)
        if fnd.all_fourier(own) and fnd.all_fourier(self._grid) and own.box == self._grid.box:
            # one forward transform per unknown, one inverse per leaf, padding folded into it
            ck = ("coeffs", model.layer_id)
            ch = self._leaves.get(ck)
            if ch is None:
                ch = self._leaves[ck] = fnd.coeffs(own, v)
            if any(alpha):
                ch = ch * jnp.asarray(fnd.symbol(own, alpha))[..., None]
            v = fnd.values(self._grid, fnd.resize(ch, own, self._grid))
        else:
            if any(alpha):
                v = own.diff(v, alpha)
            v = own.to(v, self._grid)
        out = v.reshape((self._grid.size,) + (vshape if vshape else (1,)))
        self._leaves[key] = out
        return out

    # derivatives --------------------------------------------------------------------------------------------
    def _chain(self, node):
        return derivative_chain(node, self._grid.ndim)

    @staticmethod
    def _axis(var):
        return _var_axis(var)

    def _check_family(self, expr):
        from ..utils.schemes import scheme_family

        fam = scheme_family(expr.scheme) if expr.scheme is not None else FAMILY
        if fam not in (FAMILY, "spectral"):
            raise ValueError(
                f"jno.spectral: a derivative of a spectral problem was asked with scheme={str(expr.scheme)!r}; a "
                "spectral unknown is differentiated with its own basis. Drop scheme= (or pass jno.sp(...) options)."
            )

    def _diff(self, target, alpha, ctx):
        chain = self._chain(target)
        if chain is not None:
            model, a0, comp = chain
            leaf = self._leaf(model, tuple(x + y for x, y in zip(a0, alpha)))
            return leaf if comp is None else leaf[..., comp : comp + 1 if comp != -1 else None]
        finer = self._finer_grid_of(target)
        if finer is not None:
            # a composite of unknowns on a finer grid than this one (continuity on the pressure nodes): formed and
            # differentiated where the unknowns live, then moved here -- the interpolant of the derivative
            sub_ev = _SpectralEvaluator(self.params, self._problem, finer)
            t = ctx.context.get("__time__")
            t = None if t is None else jnp.ravel(jnp.asarray(t))[0]
            sub_ctx = self._problem._context(finer, t, base=self._grid)
            vals = jnp.asarray(sub_ev.evaluate(_unwrap(target), context=sub_ctx, var_bindings={}))
            vals = jnp.broadcast_to(vals, (finer.size,) + vals.shape[1:]) if vals.shape[:1] != (finer.size,) else vals
            d = finer.diff(vals.reshape(finer.size, -1), alpha)
            return finer.to(d, self._grid).reshape((self._grid.size,) + vals.shape[1:])
        vals = jnp.asarray(self._dispatch(target, ctx))
        m = self._grid.size
        if vals.ndim == 0 or vals.shape[0] != m:
            vals = jnp.broadcast_to(vals, (m,) + vals.shape[1:] if vals.ndim else (m, 1))
        return self._grid.diff(vals.reshape(m, -1), alpha).reshape(vals.shape)

    def _finer_grid_of(self, target):
        """The one grid every spectral unknown in ``target`` lives on, when it has more nodes than this grid."""
        grids = {self._problem._grid_of(m) for m in _models_in(target)}
        if len(grids) != 1:
            return None
        g = grids.pop()
        return g if g != self._grid and g.size > self._grid.size and g.box == self._grid.box else None

    def _unit(self, ax):
        a = [0] * self._grid.ndim
        a[ax] += 1
        return a

    def _eval_jacobian(self, expr, ctx):
        self._check_family(expr)
        comps = []
        for var in expr.variables:
            tag = str(getattr(var, "tag", ""))
            if getattr(var, "axis", "spatial") == "temporal":
                raise NotImplementedError("jno.spectral: a Jacobian in the time variable; write u.t.")
            if tag.startswith("n_"):  # a normal derivative n·∇(target)
                nrm = jnp.asarray(ctx.context[tag])
                parts = [self._diff(expr.target, self._unit(i), ctx) for i in range(self._grid.ndim)]
                dn = sum(nrm[:, i].reshape((-1,) + (1,) * (parts[i].ndim - 1)) * parts[i] for i in range(len(parts)))
                comps.append(dn)
                continue
            comps.append(self._diff(expr.target, self._unit(int(var.dim[0])), ctx))
        if len(comps) == 1:
            return comps[0]
        # a scalar field's gradient is (m, n_vars), a vector field's (m, c, n_vars): jNO's convention
        comps = [c[:, 0] if c.ndim == 2 and c.shape[1] == 1 else c for c in comps]
        return jnp.stack(comps, axis=-1)

    def _eval_hessian(self, expr, ctx):
        self._check_family(expr)
        axes = [self._axis(v) for v in expr.variables]
        if None in axes:
            raise NotImplementedError("jno.spectral: a second derivative in time or along a normal.")
        if expr.trace or len(axes) == 1:
            out = None
            for ax in axes:
                a = [0] * self._grid.ndim
                a[ax] += 2
                term = self._diff(expr.target, a, ctx)
                out = term if out is None else out + term
            return out
        if len(axes) == 2:
            a = [0] * self._grid.ndim
            a[axes[0]] += 1
            a[axes[1]] += 1
            return self._diff(expr.target, a, ctx)
        rows = []
        for ai in axes:
            row = []
            for aj in axes:
                a = [0] * self._grid.ndim
                a[ai] += 1
                a[aj] += 1
                row.append(self._diff(expr.target, a, ctx))
            rows.append(jnp.stack(row, axis=-1))
        return jnp.stack(rows, axis=-2)

    def _eval_temporal_derivative(self, expr, ctx):
        raise NotImplementedError(
            "jno.spectral: a time derivative reached the spatial evaluation; it is handled by the march "
            "(write u.t in an equation with an initial condition on a time domain)."
        )


class _TraceSpectral(StrongFormProblem):
    """The spectral problem (see :func:`spectral`)."""

    _who = "jno.spectral"

    def __init__(self, constraints):
        constraints = list(constraints)
        self._constraints = constraints
        self.unknowns = _find_unknowns(constraints, who=self._who)
        plain = [u for u in self.unknowns if not _is_spectral_model(u)]
        if plain:
            raise NotImplementedError(
                "jno.spectral([...]): every unknown must be spectral (domain.unknown(space='spectral', ...)); a P1 "
                "nodal unknown is solved by jno.fdm. Couple the two problems through jno.core([...])."
            )
        doms = {id(u._fem_field_domain) for u in self.unknowns}
        if len(doms) != 1:
            raise NotImplementedError(
                "jno.spectral([...]): the unknowns live on different domains; couple them with jno.core."
            )
        self.domain = self.unknowns[0]._fem_field_domain
        self._N = -1  # no single node count: every unknown has its own (see _layout)
        self._gauges = [c for c in constraints if _is_global(c)]
        rest = [c for c in constraints if not _is_global(c)]
        self._classify(rest)
        self._resolve_families(rest)
        self._layout()
        self._build_rows()

    # classification -------------------------------------------------------------------------------------------
    def _registry(self):
        return self.domain.__dict__.get("_spectral_tags", {})

    def _region_of(self, tag):
        reg = self._registry().get(tag)
        return None if reg is None else reg[0]

    def _is_boundary_derivative_condition(self, c):
        from .._strong_form import _has_unknown_derivative

        tags = _spatial_tags(c)
        faces = {"boundary"} | {n for pair in FACE_NAMES.get(len(self.unknowns[0]._spectral_space.box), ()) for n in pair}
        return (
            bool(tags)
            and all(self._region_of(t) in faces or str(t).startswith("n_") for t in tags)
            and any(_has_unknown_derivative(c, u) for u in self.unknowns)
        )

    def _resolve_families(self, constraints):
        """Fix every unknown's families from the written ties: a tie between the two faces of an axis makes that
        axis Fourier for the unknowns the tie names."""
        tied = {u.layer_id: set() for u in self.unknowns}
        ndim = len(self.unknowns[0]._spectral_space.box)
        for c in constraints:
            pair = getattr(c, "_periodic_tie", None)
            if pair is None:
                continue
            regs = {self._region_of(t) for t in pair}
            axis = next((i for i, (lo, hi) in enumerate(FACE_NAMES.get(ndim, ())) if regs == {lo, hi}), None)
            if axis is None:
                raise ValueError(
                    f"jno.spectral([...]): the periodic tie between {pair} does not join the two faces of one axis "
                    f"(regions {sorted(map(str, regs))}); tie left-right, bottom-top or front-back."
                )
            for m in _models_in(c):
                tied[m.layer_id].add(axis)
        for u in self.unknowns:
            u._spectral_space.resolve(tied[u.layer_id])

    def _grid_of(self, model):
        return model._spectral_space.grid

    def _tag_grid(self, tag):
        reg = self._registry().get(tag)
        if reg is None:
            raise ValueError(
                f"jno.spectral([...]): the coordinates tagged {tag!r} are not points of a spectral space; take them "
                "from domain.variable(...) of the domain that carries the spectral unknowns."
            )
        return reg[1].grid

    def _grid_of_expr(self, c, default=None):
        tags = {t for t in _spatial_tags(c) if not str(t).startswith("n_")}
        grids = {self._tag_grid(t) for t in tags}
        if len(grids) > 1:
            raise NotImplementedError(
                f"jno.spectral([...]): one condition reads coordinates of {len(grids)} different node sets ({sorted(tags)}); "
                "write it on one node set."
            )
        return next(iter(grids)) if grids else default

    def _field_of(self, c):
        models = _models_in(c)
        if not models:
            raise ValueError("jno.spectral([...]): a condition names no spectral unknown.")
        lids = [u.layer_id for u in self.unknowns]
        return min(lids.index(m.layer_id) for m in models)

    # layout -----------------------------------------------------------------------------------------------------
    def _layout(self):
        self._sizes = [self._grid_of(u).size for u in self.unknowns]
        self._ncomp = [int(math.prod(np.shape(u.module.value)[1:])) for u in self.unknowns]
        self._offsets = list(np.cumsum([0] + [n * c for n, c in zip(self._sizes, self._ncomp)])[:-1])
        self._Ntot = int(sum(n * c for n, c in zip(self._sizes, self._ncomp)))
        self._nf = sum(self._ncomp)

    def _inject(self, dofs):
        import equinox as eqx

        self._require_real(dofs, "the state (an initial guess x0?)")
        out = {}
        for u, o, n, c in zip(self.unknowns, self._offsets, self._sizes, self._ncomp):
            sl = dofs[o : o + n * c]
            value = sl.reshape(c, n).T.reshape(u.module.value.shape)
            out[u.layer_id] = eqx.tree_at(lambda m: m.value, u.module, value.astype(u.module.value.dtype))
        return out

    def _unpack(self, x):
        """The solution per unknown in its value shape (``(n,)`` or ``(n, c)``); a tuple for several unknowns."""
        x = jnp.asarray(x)
        lead = x.shape[:-1]
        outs = []
        for u, o, n, c in zip(self.unknowns, self._offsets, self._sizes, self._ncomp):
            sl = x[..., o : o + n * c].reshape(lead + (c, n))
            sl = jnp.swapaxes(sl, -1, -2).reshape(lead + tuple(np.shape(u.module.value)))
            outs.append(sl)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def _pack(self, values):
        vals = values if isinstance(values, (tuple, list)) else (values,)
        parts = []
        for v, n, c in zip(vals, self._sizes, self._ncomp):
            v = jnp.asarray(v).reshape(n, c)
            parts.append(v.T.reshape(-1))
        return jnp.concatenate(parts)

    # rows -------------------------------------------------------------------------------------------------------
    def _build_rows(self):
        """Per unknown: its equation and row grid, the boundary rows (region node indices) and the gauge rows."""
        if len(self._pde) != len(self.unknowns):
            raise ValueError(
                f"{self._who}([...]): {len(self.unknowns)} unknowns need {len(self.unknowns)} equations, one each, in "
                f"the order the unknowns are declared; got {len(self._pde)}."
            )
        self._rows = []
        bcs = {k: [] for k in range(len(self.unknowns))}
        for kind, lst in (("neumann", self._neumann), ("dirichlet", self._dirichlet)):
            for c in lst:
                bcs[self._field_of(c)].append((kind, c))
        gauges = {k: [] for k in range(len(self.unknowns))}
        for c in self._gauges:
            gauges[self._field_of(c)].append(c)
        for k, (u, eq) in enumerate(zip(self.unknowns, self._pde)):
            own = self._grid_of(u)
            rowgrid = self._grid_of_expr(eq, own)
            entries = []
            for kind, c in bcs[k]:
                g = self._grid_of_expr(c, own)
                if g != own:
                    raise NotImplementedError(
                        f"{self._who}([...]): a boundary condition of unknown {k} is written on another node set; write "
                        "it with the unknown's own nodes (domain.variable(region) / nodes=U)."
                    )
                region = self._region_of(_region_tag(c))
                found = region_points(own, region)
                if found is None or found[1] is None:
                    raise ValueError(
                        f"{self._who}([...]): no nodes of unknown {k} on region {region!r} for a boundary row."
                    )
                entries.append((kind, c, np.asarray(found[1], dtype=int)))
            replace = rowgrid == own
            n_rows = (rowgrid.size if replace else rowgrid.size + sum(e[2].size for e in entries)) + (
                0 if replace else len(gauges[k])
            )
            if n_rows != own.size:
                raise ValueError(
                    f"{self._who}([...]): unknown {k} has {own.size} nodes but its equation and conditions give {n_rows} "
                    "rows; a node set other than the unknown's own needs exactly as many rows as nodes."
                )
            taken = np.zeros(own.size, dtype=bool)
            for _kind, _c, idx in entries:
                taken[idx] = True
            gauge_rows = []
            for c in gauges[k]:
                free = np.flatnonzero(~taken)
                if not free.size:
                    raise ValueError(f"{self._who}([...]): no row of unknown {k} is left for its gauge condition.")
                gauge_rows.append((c, int(free[0])))
                taken[free[0]] = True
            self._rows.append(
                {"eq": eq, "grid": rowgrid, "own": own, "bcs": entries, "gauges": gauge_rows, "replace": replace}
            )

    # evaluation --------------------------------------------------------------------------------------------------
    def _context(self, grid, t, base=None):
        """Points (and normals) of every tag on ``base`` (default ``grid``), taken at ``grid``'s nodes -- a padded
        grid reads the tags of the grid it pads."""
        base = grid if base is None else base
        ctx = {}
        for tag, (region, space) in self._registry().items():
            try:
                g = space.grid
            except RuntimeError:
                continue
            if g != base:
                continue
            ctx[tag] = jnp.asarray(grid.points)
            found = region_points(grid, region)
            if found is not None and found[2] is not None:
                nrm = np.zeros((grid.size, grid.ndim))
                nrm[found[1]] = found[2]
                ctx[f"n_{tag}"] = jnp.asarray(nrm)
        ctx["__time__"] = jnp.full((grid.size, 1), 0.0 if t is None else t)
        return ctx

    def _evaluate_on(self, expr, grid, params, t, cache, base=None, seeded=None):
        ev = cache.get(grid)
        if ev is None:
            ev = cache[grid] = (_SpectralEvaluator(params, self, grid, seeded), self._context(grid, t, base))
        evaluator, ctx = ev
        if t is not None:
            ctx = {**ctx, "__time__": jnp.full((grid.size, 1), t)}
        val = jnp.asarray(evaluator.evaluate(_unwrap(expr), context=ctx, var_bindings={}))
        if val.ndim == 0 or val.shape[0] != grid.size:
            val = jnp.broadcast_to(val.reshape(-1)[:1] if val.size == 1 else val, (grid.size,) + val.shape[1:])
        return val.reshape(grid.size, -1)

    def _equation_values(self, row, expr, params, t, cache):
        grid = row["grid"]
        a = _dealias_of(expr)
        if a is None or a <= 1.0:
            return self._evaluate_on(expr, grid, params, t, cache)
        padded = grid.with_orders(tuple(int(math.ceil(a * n)) for n in grid.shape))
        vals = self._evaluate_on(expr, padded, params, t, cache, base=grid)
        return project(padded, vals, grid)

    def _equation_coeffs(self, row, expr, params, t, cache, seeded=None):
        """An equation's value as coefficients of an all-Fourier row grid (:mod:`fourier_nd` layout): evaluated on
        the padded grid when it asks for dealiasing, and truncated there -- no inverse transform back."""
        grid = row["grid"]
        a = _dealias_of(expr)
        if a is None or a <= 1.0:
            return fnd.coeffs(grid, self._evaluate_on(expr, grid, params, t, cache, seeded=seeded))
        padded = grid.with_orders(tuple(int(math.ceil(a * n)) for n in grid.shape))
        vals = self._evaluate_on(expr, padded, params, t, cache, base=grid, seeded=seeded)
        return fnd.resize(fnd.coeffs(padded, vals), padded, grid)

    def _residual_fn(self, *, spatial=False, extra_params=None):
        """``R(dofs, t)`` over the blocked DOF vector: per unknown its equation's rows with the boundary and gauge
        rows in place. ``spatial=True`` drops the ``u.t`` terms (the method-of-lines spatial residual)."""
        scope = self._params_scope(extra_params)
        rows = [{**r, "eq": _zero_temporal(_unwrap(r["eq"])) if spatial else r["eq"]} for r in self._rows]

        def residual(dofs, t=None):
            dofs = jnp.asarray(dofs)
            params = {**scope, **self._live_params(), **self._inject(dofs)}
            cache = {}
            out = []
            for r in rows:
                vals = self._equation_values(r, r["eq"], params, t, cache)
                c = vals.shape[1]
                if r["replace"]:
                    for kind, cexpr, idx in sorted(r["bcs"], key=lambda e: e[0] == "dirichlet"):
                        bv = self._evaluate_on(cexpr, r["own"], params, t, cache)[idx]
                        vals = vals.at[jnp.asarray(idx)].set(jnp.broadcast_to(bv, (idx.size, c)))
                    for gexpr, i in r["gauges"]:
                        gv = self._evaluate_on(gexpr, r["own"], params, t, cache).reshape(-1)[:c]
                        vals = vals.at[i].set(gv)
                    out.append(vals.T.reshape(-1))
                else:
                    parts = [vals.T.reshape(-1)]
                    for kind, cexpr, idx in r["bcs"]:
                        bv = self._evaluate_on(cexpr, r["own"], params, t, cache)[idx]
                        parts.append(bv.T.reshape(-1))
                    for gexpr, _i in r["gauges"]:
                        parts.append(self._evaluate_on(gexpr, r["own"], params, t, cache).reshape(-1)[:c])
                    out.append(jnp.concatenate(parts))
            res = out[0] if len(out) == 1 else jnp.concatenate(out)
            return self._require_real(res, "the residual (a complex coefficient or source)")

        return residual

    # solves -------------------------------------------------------------------------------------------------------
    def _linear_solve(self, linear, precond=None):
        """``(mv, rhs) -> x`` for the inner solves: dense LU when small (or asked), else GMRES."""
        from .. import solve as _solve
        from ..utils.solver.solver_api import LinearOperator

        n = self._Ntot
        dense = (linear is None and precond is None and n <= _DENSE_MAX) or bool(getattr(linear, "direct", False))
        if dense:

            def solve_dense(mv, rhs):
                A = _assemble(mv, n, rhs.dtype)
                return jnp.linalg.solve(A, rhs)

            return solve_dense
        # spectral accuracy needs a spectral tolerance: GMRES's 1e-8 default would cap the solve far above it
        spec = linear if linear is not None else _solve.gmres(tol=1e-12, restart=50)
        if precond is None and linear is None:
            from .precond import SpectralPrecond

            precond = SpectralPrecond()  # the default for a large problem: the Fourier-mode block inverse
        if precond is None:
            return lambda mv, rhs: spec(LinearOperator.from_matvec(mv), rhs)
        from ..fdm import _gmres_incremental
        from .precond import materialize_for

        def solve_preconditioned(mv, rhs):
            try:
                M = materialize_for(precond, self, mv)
            except NotImplementedError:
                if linear is None and precond is not None and type(precond).__name__ == "SpectralPrecond":
                    return spec(LinearOperator.from_matvec(mv), rhs)  # the automatic choice does not apply here
                raise
            return _gmres_incremental(mv, rhs, M, spec)[0]

        return solve_preconditioned

    def solve(
        self,
        nonlinear=None,
        x0=None,
        profile=False,
        time=None,
        *,
        linear=None,
        precond=None,
        save_ts=None,
        span=None,
        log=None,
    ):
        """Solve the problem; returns each unknown's nodal values (``(n,)`` or ``(n, c)``; a tuple for several).

        Args:
            nonlinear: the Newton spec (``jno.solve.newton(...)``) for a nonlinear residual.
            x0: initial guess (nodal values per unknown, as returned) or the transient start state.
            time: a ``jno.solve`` time scheme for a transient problem (default backward Euler).
            linear: the inner linear solve: ``jno.solve.lu()`` / ``dense()`` (assembled) or ``gmres()`` (matrix-free);
                default dense up to 4096 DOFs, GMRES above.
            precond: ``jno.precond.spectral()`` (fast diagonalisation) for the matrix-free solve.
            save_ts: transient save times.
        """
        if self._transient:
            from .march import march

            return march(self, time=time, x0=x0, linear=linear, precond=precond, save_ts=save_ts, nonlinear=nonlinear)
        return self._unpack(self._steady_solve(nonlinear=nonlinear, x0=x0, linear=linear, precond=precond))

    def _data_fingerprint(self):
        from ..fdm import _TraceFDM

        return _TraceFDM._data_fingerprint(self)

    # what jno.core's coupling reads of a strong-form problem (see jno._coupling.participant) ---------------------
    def _march(self, *, time=None, x0=None, span=None, linear=None, precond=None, nonlinear=None, save_ts=None):
        from .march import march

        return march(
            self,
            time=time,
            x0=x0,
            span=span,
            linear=linear,
            precond=precond,
            nonlinear=nonlinear,
            save_ts=save_ts,
            raw=True,
        )

    def _initial_state(self):
        from .march import initial_state

        return initial_state(self)

    def _nodal_module(self, module):
        """A per-node data field on one of the problem's grids, ``(n,)``, as ``(n, 1)`` (the layout of every other
        per-node value in an evaluation)."""
        value = getattr(module, "value", None)
        sizes = set(getattr(self, "_sizes", ()) or ())
        if value is None or np.ndim(value) != 1 or np.shape(value)[0] not in sizes:
            return module
        import equinox as eqx

        with jax.ensure_compile_time_eval():
            value = jnp.reshape(value, (-1, 1))
        return eqx.tree_at(lambda m: m.value, module, value)

    def _steady_solve(self, nonlinear=None, x0=None, linear=None, precond=None, extra_params=None):
        from .. import solve as _solve
        from ..fdm import _fd_newton_tolerances

        n = self._Ntot
        if extra_params:
            residual = jax.jit(self._residual_fn(extra_params=extra_params))
        else:
            # compiled once per problem and data: a repeated solve (a coupling sweep, a sweep over x0) reuses it; a
            # changed data field or network (the values a compiled residual bakes in) recompiles
            cache = self.__dict__.setdefault("_steady_residual_jit", {})
            key = self._data_fingerprint()
            residual = cache.get(key)
            if residual is None:
                residual = cache[key] = jax.jit(self._residual_fn())
        u0 = jnp.zeros(n) if x0 is None else (jnp.asarray(x0) if np.ndim(x0) == 1 and np.size(x0) == n else self._pack(x0))
        ls = self._linear_solve(linear, precond)
        # affinity is structural (data and parameter VALUES do not change it): decided once per problem
        if self._is_affine("steady", lambda: residual, n):
            r0 = residual(jnp.zeros(n))
            mv = lambda v: jax.jvp(residual, (jnp.zeros(n),), (v,))[1]  # noqa: E731
            return ls(mv, -r0)
        driver = nonlinear or _solve.newton(**_fd_newton_tolerances(residual, u0))
        return driver(residual, u0, linear_solve=ls)


def _assemble(mv, n, dtype, batch=256):
    """The dense matrix of a linear map by ``n`` JVP columns, ``batch`` at a time (memory ``batch · n``)."""
    eye_cols = jnp.arange(n)

    def cols(idx):
        E = jax.nn.one_hot(idx, n, dtype=dtype)
        return jax.vmap(mv)(E)

    pad = (-n) % batch
    idx = jnp.concatenate([eye_cols, jnp.zeros(pad, dtype=eye_cols.dtype)]).reshape(-1, batch)
    blocks = jax.lax.map(cols, idx)  # (nb, batch, n): rows are A e_j
    return blocks.reshape(-1, n)[:n].T


def project(src, values, dst):
    """Values on grid ``dst`` of the spectral projection of a field on the finer grid ``src``: the Fourier spectrum
    truncated, the polynomial modal coefficients truncated -- the projection that makes padding remove aliasing."""
    from .bases import Fourier

    if fnd.all_fourier(src) and fnd.all_fourier(dst):
        return fnd.values(dst, fnd.resize(fnd.coeffs(src, values), src, dst))
    u = src.as_grid(values)
    for i, (a, b) in enumerate(zip(src.axes, dst.axes)):
        if isinstance(a, Fourier):
            u = a.resample(u, b.n, axis=i)
        else:
            c = a.forward(u, axis=i)
            c = jnp.take(c, jnp.arange(b.n), axis=i)
            u = b.backward(c, axis=i)
    return dst.as_flat(u)


def spectral(constraints):
    """Spectral collocation of a strong-form constraint list -- the spectral sibling of :func:`jno.fdm`.

    The unknowns are spectral (``domain.unknown(space="spectral", order=(nx, ny))``); the constraints are written
    as for ``jno.fdm``. A written periodic tie ``U(xl, yl) - U(xr, yr)`` makes that axis Fourier; every other
    axis is Chebyshev (``space=("fourier", "legendre")`` names them). Example::

        dom = jno.shape.rect(0, -1, 2 * pi, 1).domain()
        U = dom.unknown(space="spectral", order=(32, 25))
        x, y, _ = dom.variable("interior", split=True)
        xl, yl, _ = dom.variable("left", split=True); xr, yr, _ = dom.variable("right", split=True)
        xb, yb, _ = dom.variable("bottom", split=True); xt, yt, _ = dom.variable("top", split=True)
        u = U.bind(x=x, y=y)
        sol = jno.spectral([-(u.xx + u.yy) - f, U(xb, yb) - 0.0, U(xt, yt) - 0.0, U(xl, yl) - U(xr, yr)]).solve()
        coeffs = U.basis.coeffs(sol)

    Args:
        constraints: the equations (one per unknown, in declaration order), boundary rows, initial conditions,
            periodic ties and gauges (``psi.mean``).

    Returns:
        The problem; call ``.solve()``.
    """
    if not isinstance(constraints, (list, tuple)):
        raise TypeError(f"jno.spectral expects a constraint LIST; got {type(constraints).__name__}.")
    return _TraceSpectral(list(constraints))
