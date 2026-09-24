"""Slip normals that follow runtime (``.trainable()``) mesh coordinates.

The exact slip elimination ``n·u = 0`` carries ``u = P ũ``, where the ENTRIES of ``P`` come from the
per-node normals ``N_i = ∫ φ_i n ds`` of the slip surface (``_fem._region_node_normals``). When that
surface's vertices are runtime coordinates, the normals move with them, so ``P`` must be rebuilt from the
coordinates the solve is actually using -- a ``P`` baked at build enforces the OLD surface's condition on
the new mesh and returns a converged, wrong answer (measured 3e-4 relative on a 3-D rolling model).

What stays fixed is everything STRUCTURAL: which nodes are constrained, which component each eliminates
(the build-time pivot), the reduced column of every free component, which facets make up the surface and
their outward orientation, and which normal RULE each node uses (flux-exact mass normal; angle-weighted
average for a flux-neutral P2 vertex; area-weighted in 1-D/2-D). Only VALUES change, so ``P`` keeps one
sparsity and the runtime version is pure JAX and differentiable in the coordinates.

Scope: the steady nonlinear path (``fem.solve(param=...)``, continuation, the convergence verdict). The
build-time ``P`` of such a reduction is replaced by :class:`StaleSlipP`, which raises on any use -- so a
consumer that was not taught to bind the runtime ``P`` fails loudly instead of silently using stale
normals.
"""

from __future__ import annotations

from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse


class StaleSlipP:
    """Placeholder for a slip prolongation whose values depend on runtime coordinates.

    Carries the (structural, always valid) ``shape``; ANY other use raises, naming the fix. It exists so
    that a code path which reads ``P`` without binding the solve's coordinates cannot silently use the
    build-time normals."""

    def __init__(self, shape):
        self.shape = tuple(int(s) for s in shape)
        self.ndim = 2

    def _refuse(self, *_a, **_k):
        raise NotImplementedError(
            "jno.fem: this slip condition's normals depend on runtime (trainable) mesh coordinates, so its "
            "prolongation P must be rebuilt from the coordinates of each solve, and this code path uses P "
            "without them. Supported: the steady nonlinear solve (fem.solve(param=...), continuation). For "
            "any other route rebuild jno.fem on the moved mesh, or keep the slip surface's vertices fixed."
        )

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        self._refuse()

    __matmul__ = __rmatmul__ = __array__ = __jax_array__ = _refuse

    def __repr__(self):
        return f"StaleSlipP(shape={self.shape})"


# --------------------------------------------------------------------------------------------------
# the normal rule, written once over arrays -- numpy at build (to check it against the reference
# implementation), jax.numpy at solve time
# --------------------------------------------------------------------------------------------------
def _region_vectors(xp, X, rows, sign, dim, order, n_loc):
    """Per-local-node ``(mass, area, angle)`` vectors of one region, from facet rows over local points ``X``."""
    corners = X[rows[:, :dim]]  # (nf, dim, dim)
    if dim == 3:
        cn = 0.5 * xp.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    elif dim == 2:
        e = corners[:, 1] - corners[:, 0]
        cn = xp.stack([e[:, 1], -e[:, 0]], axis=1)  # |cn| = length; orientation fixed by `sign`
    else:
        raise NotImplementedError("runtime slip normals: 1-D slip surfaces are points; nothing moves them")
    cn = cn * sign[:, None]  # = A_f * n_f, outward
    extras = rows[:, dim:]

    def scatter(idx, vals):
        idx = idx.reshape(-1)
        vals = vals.reshape(idx.shape[0], dim)
        if xp is np:
            out = np.zeros((n_loc, dim))
            np.add.at(out, idx, vals)
            return out
        return jax.ops.segment_sum(vals, jnp.asarray(idx), num_segments=n_loc)

    nf = rows.shape[0]
    rep = lambda a, k: xp.broadcast_to(a[:, None, :], (nf, k, dim))  # noqa: E731
    if order == 1:
        mass = scatter(rows[:, :dim], rep(cn / dim, dim))
    elif dim == 3:  # P2 triangle: ∫φ_vertex = 0, ∫φ_midside = A/3
        mass = scatter(extras, rep(cn / 3.0, extras.shape[1]))
    else:  # P2 line: ∫φ_end = L/6, ∫φ_mid = 2L/3
        mass = scatter(rows[:, :dim], rep(cn / 6.0, dim)) + scatter(extras, rep(2.0 * cn / 3.0, extras.shape[1]))
    area = scatter(rows, rep(cn, rows.shape[1]))
    if dim == 3:
        A = xp.linalg.norm(cn, axis=1)
        n_f = cn / A[:, None]
        th = []
        for a in range(3):
            e1 = corners[:, (a + 1) % 3] - corners[:, a]
            e2 = corners[:, (a + 2) % 3] - corners[:, a]
            c = xp.sum(e1 * e2, axis=1) / (xp.linalg.norm(e1, axis=1) * xp.linalg.norm(e2, axis=1))
            th.append(xp.arccos(xp.clip(c, -1.0, 1.0)))
        angw = scatter(rows[:, :3], xp.stack(th, axis=1)[:, :, None] * n_f[:, None, :])
    else:
        angw = None
    return mass, area, angw


def _pick(xp, mass, area, angw, nodes, rule):
    """Unit normal per constrained node by the build-time rule: 0 = mass, 1 = angle, 2 = area."""
    m, a = mass[nodes], area[nodes]
    g = angw[nodes] if angw is not None else a
    r = rule[:, None]
    v = xp.where(r == 0, m, xp.where(r == 1, g, a))
    return v / xp.linalg.norm(v, axis=1, keepdims=True)


# --------------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------------
def build_plan(domain, pts_i, cells_i, order_i, regions, nodes, node_dofs, reference_normals, kept, n_i, n_red):
    """Everything static that the runtime P needs, checked against the build-time normals.

    ``reference_normals[r]`` is ``_region_node_normals`` for ``regions[r]`` at build -- the plan's own
    formula must reproduce it to round-off before it is trusted with moved coordinates."""
    from ..._fem import _boundary_facets, _face_nodes
    from .fem_facets import build_facet_connectivity, compute_face_normals

    pts_i = np.asarray(pts_i, dtype=float)
    dim = int(pts_i.shape[1])
    V0 = np.asarray(getattr(domain, "_fem_assembly_points", None), dtype=float)
    if V0 is None or V0.ndim != 2:
        raise NotImplementedError("runtime slip normals: the assembly (P1 geometry) points are unavailable")
    V0 = V0[:, :dim]
    specs = []
    for e in getattr(domain, "_trainable_coords", None) or []:
        specs.append((np.asarray(e["ids"], dtype=np.int64), int(e["axis"]), str(e["name"])))

    cells_i = np.asarray(cells_i)
    facets = _boundary_facets(pts_i, cells_i, dim, int(order_i))
    ctype = "tetrahedron" if dim == 3 else "triangle"
    conn = build_facet_connectivity(cells_i, ctype)
    fn = np.asarray(compute_face_normals(pts_i, conn, cells_i, ctype))
    oriented = {frozenset(int(v) for v in row): fn[k] for k, row in enumerate(np.asarray(conn.face_nodes))}
    bnodes = np.unique(facets)

    reg_rows, reg_sign = [], []
    for region in regions:
        sel = _face_nodes(domain, pts_i, bnodes, region)
        in_region = np.zeros(pts_i.shape[0], dtype=bool)
        in_region[np.asarray(sel, dtype=int)] = True
        rows, sgn = [], []
        for row in facets:
            verts = [int(v) for v in row[:dim]]
            if not all(in_region[v] for v in verts):
                continue
            n_f = oriented.get(frozenset(verts))
            if n_f is None:
                continue
            P = pts_i[verts]
            if dim == 3:
                c = 0.5 * np.cross(P[1] - P[0], P[2] - P[0])
            else:
                e = P[1] - P[0]
                c = np.array([e[1], -e[0]])
            if np.linalg.norm(c) <= 0.0:
                continue
            rows.append(np.asarray(row, dtype=np.int64))
            sgn.append(1.0 if float(np.dot(c, n_f[:dim])) > 0.0 else -1.0)
        reg_rows.append(np.stack(rows))
        reg_sign.append(np.asarray(sgn))

    # local point set = every point any slip facet touches; each is a vertex or a straight midside node
    loc = np.unique(np.concatenate([r.reshape(-1) for r in reg_rows]))
    g2l = {int(g): k for k, g in enumerate(loc)}
    scale = max(float(np.ptp(V0)), 1e-300)
    vkey = {tuple(np.round(p / scale, 11)): j for j, p in enumerate(V0)}
    src = np.zeros((loc.size, 2), dtype=np.int64)
    known = np.zeros(loc.size, dtype=bool)
    for k, g in enumerate(loc):
        j = vkey.get(tuple(np.round(pts_i[g] / scale, 11)))
        if j is not None:
            src[k] = (j, j)
            known[k] = True
    for rows in reg_rows:  # midside nodes: the pair of facet corners whose midpoint they are
        for row in rows:
            cs = [int(v) for v in row[:dim]]
            for x in row[dim:]:
                k = g2l[int(x)]
                if known[k]:
                    continue
                for a in range(dim):
                    for b in range(a + 1, dim):
                        if np.allclose(0.5 * (pts_i[cs[a]] + pts_i[cs[b]]), pts_i[int(x)], atol=1e-12 * scale):
                            src[k] = (src[g2l[cs[a]]][0], src[g2l[cs[b]]][0])
                            known[k] = True
    if not known.all():
        raise NotImplementedError(
            "runtime slip normals: a slip-surface node is neither a mesh vertex nor a straight midside "
            "node, so its position cannot be followed from the vertex coordinates (curved geometry?)."
        )
    reg_rows_l = [np.vectorize(g2l.get)(r).astype(np.int64) for r in reg_rows]

    # the rule each node uses, decided by the build-time values -- and the formula checked
    Xl = 0.5 * (V0[src[:, 0]] + V0[src[:, 1]])
    node_rows: List[List[tuple]] = [[] for _ in nodes]  # per node: (region, local index, rule)
    node_pos = {int(n): a for a, n in enumerate(nodes)}
    for r, (rows, sgn) in enumerate(zip(reg_rows_l, reg_sign)):
        mass, area, angw = _region_vectors(np, Xl, rows, sgn, dim, int(order_i), loc.size)
        for g, nref in reference_normals[r].items():
            k = g2l[int(g)]
            if np.linalg.norm(mass[k]) > 1e-30:
                rule, vec = 0, mass[k]
            elif angw is not None and np.linalg.norm(angw[k]) > 1e-30:
                rule, vec = 1, angw[k]
            else:
                rule, vec = 2, area[k]
            got = vec / np.linalg.norm(vec)
            if np.max(np.abs(got - np.asarray(nref)[:dim])) > 1e-9:
                raise AssertionError(
                    f"runtime slip normals: the plan's normal at node {g} differs from the build-time one "
                    f"({got} vs {nref}); refusing to track a normal it cannot reproduce."
                )
            node_rows[node_pos[int(g)]].append((r, k, rule))

    # per node: which components the build eliminated (the pivots) and the reduced column of the rest
    kept = np.asarray(kept, dtype=np.int64)
    col_of = np.full(int(n_i), -1, dtype=np.int64)
    col_of[kept] = np.arange(kept.size)
    groups: Dict[int, dict] = {}
    for a, nd in enumerate(np.asarray(node_dofs)):
        piv = [c for c in range(dim) if col_of[nd[c]] < 0]
        free = [c for c in range(dim) if col_of[nd[c]] >= 0]
        m = len(node_rows[a])
        if len(piv) != m:
            raise AssertionError(f"runtime slip normals: node {a} eliminates {len(piv)} components for {m} conditions")
        gr = groups.setdefault(m, {"nodes": [], "piv": [], "free": [], "rows": [], "cols": [], "src": []})
        gr["nodes"].append(a)
        gr["piv"].append(piv)
        gr["free"].append(free)
        gr["src"].append([(r, k, rule) for (r, k, rule) in node_rows[a]])
        gr["rows"].append([int(nd[p]) for p in piv])
        gr["cols"].append([int(col_of[nd[f]]) for f in free])
    for gr in groups.values():
        for key in ("piv", "free", "rows", "cols", "src"):
            gr[key] = np.asarray(gr[key], dtype=np.int64)

    plan = {
        "dim": dim,
        "order": int(order_i),
        "V0": V0,
        "specs": specs,
        "src": src,
        "n_loc": int(loc.size),
        "reg_rows": reg_rows_l,
        "reg_sign": reg_sign,
        "groups": groups,
        "kept": kept,
        "shape": (int(n_i), int(n_red)),
    }
    # STRUCTURAL zeros. An entry of P that is exactly zero at the build coordinates AND at random small
    # perturbations of every trainable axis is identically zero under the allowed motion (e.g. the z-part of
    # a normal on a surface y = f(x) when only z moves), so it is left out of the pattern. Keeping it costs
    # more than an entry: when the tangent's own pattern is not concrete the sparse reduction expands every
    # triplet over D^2 slot pairs (D = entries per row of P), and one extra entry per slip row quadrupled
    # the reduced tangent a 3-D solve factorized (1.9M -> 7.6M stored, 5.5x the factorization time). Every
    # eager solve re-checks the pruned entries at its own coordinates (`check_pruned`), so a motion that
    # does make one nonzero raises instead of being dropped.
    rng = np.random.default_rng(0)
    scale = float(np.ptp(V0)) or 1.0
    samples = [None]
    for _ in range(2):
        samples.append({nm: V0[ids, ax] + 1e-4 * scale * rng.standard_normal(ids.size) for ids, ax, nm in specs})
    vals = [np.abs(_group_values(plan, a, np)) for a in samples]
    plan["mask"] = np.flatnonzero(np.maximum.reduce(vals) > 0.0)
    plan["pruned"] = np.setdiff1d(np.arange(vals[0].size), plan["mask"])
    return plan


def _group_values(plan, args, xp):
    """All (pivot, free) entries of every slip node, concatenated over the groups, before pruning."""
    return plan_P(plan, args, xp=xp, _raw=True)


def plan_P(plan, args=None, xp=jnp, _raw=False):
    """The slip prolongation for the coordinates in ``args`` (build-time coordinates when absent)."""
    dim = plan["dim"]
    V = xp.asarray(plan["V0"])
    if args is not None:
        for ids, axis, name in plan["specs"]:
            if name in args:
                val = xp.asarray(args[name]).reshape(-1)
                V = V.at[ids, axis].set(val.astype(V.dtype)) if xp is jnp else _np_set(V, ids, axis, val)
    X = 0.5 * (V[plan["src"][:, 0]] + V[plan["src"][:, 1]])
    per_region = []
    for rows, sgn in zip(plan["reg_rows"], plan["reg_sign"]):
        per_region.append(_region_vectors(xp, X, rows, xp.asarray(sgn), dim, plan["order"], plan["n_loc"]))

    kept = plan["kept"]
    idx = [np.stack([kept, np.arange(kept.size)], axis=1)]
    data = [xp.ones(kept.size)]
    for m, gr in sorted(plan["groups"].items()):
        C = []
        for j in range(m):  # j-th condition of every node in the group
            r = gr["src"][:, j, 0]
            k = gr["src"][:, j, 1]
            rule = gr["src"][:, j, 2]
            rows_j = []
            for reg in np.unique(r):
                sel = np.flatnonzero(r == reg)
                mass, area, angw = per_region[int(reg)]
                rows_j.append((sel, _pick(xp, mass, area, angw, k[sel], xp.asarray(rule[sel]))))
            Cj = xp.zeros((len(gr["nodes"]), dim))
            for sel, v in rows_j:
                Cj = Cj.at[sel].set(v) if xp is jnp else _np_rows(Cj, sel, v)
            C.append(Cj)
        C = xp.stack(C, axis=1)  # (Ng, m, dim)
        Cp = xp.take_along_axis(C, xp.asarray(gr["piv"])[:, None, :], axis=2)  # (Ng, m, m)
        Cf = xp.take_along_axis(C, xp.asarray(gr["free"])[:, None, :], axis=2)  # (Ng, m, d-m)
        R = xp.linalg.solve(Cp, Cf)  # u_piv = -R u_free
        Ng, nfree = Cf.shape[0], Cf.shape[2]
        rr = np.repeat(gr["rows"][:, :, None], nfree, axis=2)
        cc = np.repeat(gr["cols"][:, None, :], m, axis=1)
        idx.append(np.stack([rr.reshape(-1), cc.reshape(-1)], axis=1))
        data.append((-R).reshape(-1))
    if _raw:
        return xp.concatenate(data[1:]) if len(data) > 1 else xp.zeros(0)
    if "mask" in plan:  # keep the identity rows and only the structurally nonzero slip entries
        gi = np.concatenate(idx[1:]) if len(idx) > 1 else np.zeros((0, 2), np.int64)
        gd = xp.concatenate(data[1:]) if len(data) > 1 else xp.zeros(0)
        idx = [idx[0], gi[plan["mask"]]]
        data = [data[0], gd[plan["mask"]]]
    I = np.concatenate(idx).astype(np.int32)
    D = xp.concatenate(data)
    if xp is np:
        return I, D
    # The PATTERN must stay concrete under a trace: the sparse reductions read it host-side, and
    # `jnp.asarray(numpy)` inside jit is a tracer. Without this the reduction falls back to the DENSE product.
    with jax.ensure_compile_time_eval():
        Ij = jnp.asarray(I)
    return jsparse.BCOO((D, Ij), shape=plan["shape"])


def _np_set(V, ids, axis, val):
    V = np.array(V)
    V[ids, axis] = np.asarray(val)
    return V


def _np_rows(C, sel, v):
    C = np.array(C)
    C[sel] = v
    return C


# --------------------------------------------------------------------------------------------------
# solve time
# --------------------------------------------------------------------------------------------------
def is_runtime(periodic) -> bool:
    return isinstance(periodic, dict) and "slip_runtime" in periodic


def install_sentinels(periodic: dict) -> dict:
    """Replace the build-time P (and everything derived from it) by :class:`StaleSlipP`."""
    rt = periodic["slip_runtime"]
    if "blocks" in periodic:
        b = rt["block"]
        blk = dict(periodic["blocks"][b])
        blk["P"] = StaleSlipP(rt["plan"]["shape"])
        periodic["blocks"] = [blk if i == b else x for i, x in enumerate(periodic["blocks"])]
        periodic["P_blockdiag"] = StaleSlipP((int(periodic["off_full"][-1]), int(periodic["off_red"][-1])))
    else:
        periodic["P"] = StaleSlipP(rt["plan"]["shape"])
        periodic["P_node"] = periodic["P"]
    return periodic


def bind_periodic(periodic, args):
    """``periodic`` with its slip prolongation rebuilt for the coordinates in ``args``; unchanged otherwise."""
    if not is_runtime(periodic):
        return periodic
    if args is None:
        return periodic  # the sentinels raise at first use -- loudly, not with stale normals
    rt = periodic["slip_runtime"]
    P = plan_P(rt["plan"], args)
    out = dict(periodic)
    if "blocks" in periodic:
        b = rt["block"]
        blocks = [dict(x) for x in periodic["blocks"]]
        blocks[b]["P"] = P
        out["blocks"] = blocks
        off_f, off_r = np.asarray(periodic["off_full"]), np.asarray(periodic["off_red"])
        rows, cols, data = [], [], []
        for i, blk in enumerate(blocks):
            Pi = blk["P"]
            ind = np.asarray(Pi.indices) if i != b else np.asarray(rt["P_indices"])
            rows.append(ind[:, 0] + int(off_f[i]))
            cols.append(ind[:, 1] + int(off_r[i]))
            data.append(Pi.data)
        with jax.ensure_compile_time_eval():  # concrete pattern under a trace (see plan_P)
            Ibd = jnp.asarray(np.stack([np.concatenate(rows), np.concatenate(cols)], axis=1))
        out["P_blockdiag"] = jsparse.BCOO(
            (jnp.concatenate([jnp.asarray(d, P.data.dtype) for d in data]), Ibd),
            shape=(int(off_f[-1]), int(off_r[-1])),
        )
    else:
        out["P"] = P
        out["P_node"] = P
    return out


def check_pruned(periodic, args) -> None:
    """Raise if a pruned (structurally zero at build) entry of the slip P is nonzero at these coordinates.

    Eager: call it where the solve's parameter values are concrete (the verdict after a solve)."""
    if not is_runtime(periodic) or args is None:
        return
    plan = periodic["slip_runtime"]["plan"]
    if not plan.get("pruned", np.zeros(0)).size:
        return
    concrete = {k: np.asarray(v) for k, v in args.items() if not isinstance(v, jax.core.Tracer)}
    if len(concrete) != len(args):
        return  # traced (grad/jit over the solve): nothing concrete to check here
    raw = np.asarray(plan_P(plan, concrete, xp=jnp, _raw=True))
    worst = float(np.max(np.abs(raw[plan["pruned"]])))
    if worst > 1e-12:
        raise NotImplementedError(
            f"jno.fem: the runtime coordinates tilted the slip surface into a direction its build-time "
            f"pattern left out (a pruned prolongation entry is {worst:.2e}, not 0): the slip condition was "
            "NOT imposed exactly on this solve. The pattern keeps only entries that can move under the "
            "trainable axes seen at build; free the axis that produces this motion before jno.fem(...), "
            "or rebuild jno.fem on the moved mesh."
        )


def runtime_info(domain, pts_i, slip_points) -> bool:
    """Whether any trainable (runtime) vertex coordinate lies on this slip surface."""
    tc = getattr(domain, "_trainable_coords", None) or []
    if not tc:
        return False
    pts = np.asarray(domain.mesh.points, dtype=float)
    dim = int(np.asarray(slip_points).shape[1])
    scale = max(float(np.ptp(pts[:, :dim])), 1e-300)
    key = lambda a: {tuple(r) for r in np.round(np.asarray(a, dtype=float)[:, :dim] / scale, 10)}  # noqa: E731
    on = key(slip_points)
    return any(key(pts[np.asarray(e["ids"], dtype=int)]) & on for e in tc)


__all__: List[Any] = ["StaleSlipP", "bind_periodic", "build_plan", "install_sentinels", "is_runtime", "plan_P"]
