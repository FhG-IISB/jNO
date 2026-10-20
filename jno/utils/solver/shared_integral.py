"""Integrals that differ only in which known fields they read: ONE batched kernel per evaluation -- any integrand.

A matrix of integrals over a family of solutions -- an admittance, capacitance or inductance matrix
``Y_jk = ∫ a(u_j, u_k)``, the effective tensor of homogenisation from its cell problems, one functional over many
load cases -- is written as one ``expr.integrate(d)`` per entry. Each entry used to be a cell kernel of its own,
in the forward program and again in its gradient, so the program (and its compile time) grew with the number of
entries. Within one evaluation jNO now finds the integrals whose integrands have the same structure and read
fields of the same elements, and evaluates each such group as one kernel vmapped over its members.

Matching is by a structural key (:func:`operator_key.structural_key`) in which every frozen field is a numbered
SLOT carrying its element (space, order, value shape, block, solve or data); everything else keys as for the shared
factorisation -- parameters, region masks and functions by what they are, and a node type the key does not know by
identity, so a key can only MISS a grouping, never put two different integrands together. A slot that holds the
same field in every member stays unbatched; the others are stacked (a solve's values come from the evaluation's
memo, solved outside the vmap -- with the shared factorisation where it applies). Each member keeps its own cell
support (a terminal's integrand lives on its bar's cells), padded to the longest with weight-0 cells.
"""

from __future__ import annotations

import numpy as np


def _key(node):
    """``node``'s grouping key (cached on its spec); its frozen fields in slot order go to ``spec["slots"]``."""
    from ...trace import FrozenField
    from .operator_key import structural_key

    spec = node._integral_spec
    if "key" not in spec:
        slots, index = [], {}

        def leaf(n):
            if not isinstance(n, FrozenField):
                return None
            if id(n) not in index:
                index[id(n)] = len(slots)
                slots.append(n)
            blk = getattr(n, "values_block", None)
            return (
                "slot",
                index[id(n)],
                str(n.space).upper(),
                int(n.order),
                tuple(n.value_shape or ()),
                None if blk is None else tuple(blk),
                getattr(n, "values_node", None) is not None,
            )

        spec["key"] = (id(spec["domain"]), structural_key(spec["expr"], {}, leaf))
        spec["slots"] = slots
    return spec["key"]


def _members(evaluator, node, memo):
    """The integrals of this evaluation that share ``node``'s key and are not evaluated yet (``node`` first)."""
    from .shared_solve import nodes_where

    roots = getattr(evaluator, "_per_eval_roots", None)
    groups = getattr(evaluator, "_per_eval_igroups", None)
    if groups is None or groups[0] is not roots:
        groups = (roots, {})
        evaluator._per_eval_igroups = groups
        for n in nodes_where(roots or [], lambda n: getattr(n, "_integral_spec", None) is not None):
            groups[1].setdefault(_key(n), []).append(n)
    members = [n for n in groups[1].get(_key(node), []) if id(n) not in memo and n is not node]
    return [node] + members


def _plan(members):
    """Host setup of a group, once: the template functional (member 0's, with the varying data slots as inputs)
    and every member's cell support, padded to one length."""
    from .fem_known import cell_support, domain_functional, frozen_cell_rows
    from .fem_utils import _cell_region_mask

    spec0 = members[0]._integral_spec
    domain = spec0["domain"]
    slots = [m._integral_spec["slots"] for m in members]
    varying = [i for i in range(len(slots[0])) if any(s[i] is not slots[0][i] for s in slots[1:])]
    data = {slots[0][i].frozen_id for i in varying if getattr(slots[0][i], "values_node", None) is None}
    fn, inputs = domain_functional(domain, spec0["expr"], as_inputs=frozenset(data))
    n_cells = len(np.asarray(domain.mesh.cells_dict["tetra" if int(domain.dimension) == 3 else "triangle"]))
    masks: dict = {}

    def region(r):
        if r not in masks:
            masks[r] = np.asarray(_cell_region_mask(domain, r)).reshape(-1) != 0
        return masks[r]

    supports = []
    for m, s in zip(members, slots):
        where = {id(f): ("frozen", slots[0][i].frozen_id) for i, f in enumerate(s)}

        def rows(f, _where=where):
            if getattr(f, "values_node", None) is not None:
                return None
            return frozen_cell_rows(np.reshape(np.asarray(f.values), (-1,)), fn.specs[_where[id(f)]])

        sup = cell_support(m._integral_spec["expr"], n_cells, rows, region)
        supports.append(np.arange(n_cells) if sup is None else np.flatnonzero(sup))
    length = max(1, max(len(ids) for ids in supports))
    ids = np.zeros((len(members), length), np.int64)
    w = np.zeros((len(members), length))
    for k, sup in enumerate(supports):
        ids[k, : len(sup)], w[k, : len(sup)] = sup, 1.0
        if len(sup):
            ids[k, len(sup) :] = sup[0]  # a real cell, weighted 0: its value is finite and drops out
    vkeys = {("frozen", slots[0][i].frozen_id): i for i in varying}
    consts = {
        key: np.stack([np.reshape(np.asarray(s[i].values), (-1,)) for s in slots])
        for key, i in vkeys.items()
        if slots[0][i].frozen_id in data
    }
    return {"members": list(members), "fn": fn, "inputs": inputs, "ids": ids, "w": w, "vkeys": vkeys, "consts": consts}


def integrate_shared(evaluator, node, ctx, memo) -> bool:
    """Evaluate ``node`` together with every integral of the current evaluation that shares its key; fill the
    memo for all of them. ``False`` (nothing done) when it has no partner."""
    import jax
    import jax.numpy as jnp

    members = _members(evaluator, node, memo)
    if len(members) < 2:
        return False
    cache = members[0]._integral_spec.setdefault("plans", {})
    ck = tuple(id(m) for m in members)
    plan = cache.get(ck)
    if plan is None or any(a is not b for a, b in zip(plan["members"], members)):
        with jax.ensure_compile_time_eval():  # host setup: its arrays are constants, never this trace's tracers
            plan = cache[ck] = _plan(members)
    slots = [m._integral_spec["slots"] for m in members]
    fixed = {key: evaluator._dispatch(n, ctx) for key, n in plan["inputs"] if key not in plan["vkeys"]}
    batched = {key: jnp.asarray(v) for key, v in plan["consts"].items()}
    for key, i in plan["vkeys"].items():
        if key not in batched:  # a solve per member, from the memo (or the shared factorisation), outside the vmap
            batched[key] = jnp.stack(
                [jnp.reshape(jnp.asarray(evaluator._dispatch(s[i].values_node, ctx)), (-1,)) for s in slots]
            )
    fn, n = plan["fn"], len(members)
    out = jax.vmap(lambda b, ids, w: fn({**fixed, **b}, ids, w, n_batch=n))(
        batched, jnp.asarray(plan["ids"]), jnp.asarray(plan["w"])
    )
    for k, m in enumerate(members):
        memo[id(m)] = (m, out[k])
    return True
