"""One factorisation for the solves that share their operator, inside one evaluation -- any solver.

Forms that differ only in their load (one per drive, load case, incident direction, sample) have the same
operator key (:mod:`operator_key`). When an evaluation holds several such solves with the same solver spec,
the first one dispatched solves them all: every member's system is assembled (cheap next to a
factorisation), the first member's matrix takes the block of right-hand sides -- one factorisation on the
cuDSS LDL^T path (in one call, forward and adjoint) and for host SuperLU (its factorisation cache) -- and each
member's result goes into the evaluation's memo. Each column is checked against its OWN member's matrix by the residual gate, so a
key that matched forms that are not the same raises instead of returning a wrong solution.
"""

from __future__ import annotations


def _solve_nodes(roots):
    """Every solve node reachable from ``roots`` (through expression children, a frozen field's solve and a
    function call's arguments), in first-seen order."""
    return nodes_where(roots, lambda n: getattr(n, "_solve_fls", None) is not None)


def nodes_where(roots, pred):
    """Every node reachable from ``roots`` (through expression children, a frozen field's solve and a function
    call's arguments) for which ``pred`` holds, in first-seen order."""
    from .solver_helper import iter_children

    out, seen, stack = [], set(), list(reversed(list(roots)))
    while stack:
        n = stack.pop()
        if id(n) in seen or n is None:
            continue
        seen.add(id(n))
        if pred(n):
            out.append(n)
        nxt = []
        vn = getattr(n, "values_node", None)
        if vn is not None:
            nxt.append(vn)
        try:
            nxt.extend(iter_children(n))
        except Exception:  # noqa: BLE001
            pass
        for a in getattr(n, "args", None) or ():
            if hasattr(a, "__class__") and a.__class__.__module__.startswith("jno"):
                nxt.append(a)
        stack.extend(reversed(nxt))
    return out


def _group_of(node):
    """``(key, fem)`` of a solve that may share its factorisation, else ``None``: the form's operator key and the
    solver it asked for (forms share only on one solver spec)."""
    fem = getattr(node, "_fem_ref", None)
    if fem is None or getattr(node, "_solve_fn", None) is None:
        return None
    key = fem._operator_key()
    if key is None:
        return None
    spec = getattr(node, "_linear_spec", None)
    spec_key = None if spec is None else (getattr(spec, "name", None), getattr(spec, "key", None) or id(spec))
    return (key, spec_key), fem


def solve_shared(evaluator, node, ctx, memo) -> bool:
    """Solve ``node`` together with every solve of the current evaluation that shares its operator and solver;
    fill the memo for all of them. ``False`` (nothing done) when it has no partner.

    Any solver: the in-program cuDSS LDL^T path, PARDISO and the nvmath cuDSS LU take the block of right-hand
    sides in one call (one factorisation, forward and adjoint); any other solver gets the columns one by one on
    the same matrix (host SuperLU reuses its cached factorisation; cuSolver and Krylov solve per column, as
    separate solves would)."""
    import jax.numpy as jnp

    from .solver_api import residual_gate

    g = _group_of(node)
    if g is None:
        return False
    groups = getattr(evaluator, "_per_eval_groups", None)
    if groups is None or groups[0] is not getattr(evaluator, "_per_eval_roots", None):
        groups = (getattr(evaluator, "_per_eval_roots", None), {})
        evaluator._per_eval_groups = groups
        for n in _solve_nodes(groups[0] or []):
            gn = _group_of(n)
            if gn is not None:
                groups[1].setdefault(gn[0], []).append(n)
    members = [n for n in groups[1].get(g[0], []) if id(n) not in memo]
    if node not in members:
        members = [node] + members
    if len(members) < 2:
        return False
    systems, rhs = [], []
    for n in members:
        fls = n._solve_fls
        names = list(fls.runtime_parameter_exprs)
        vals = [evaluator._dispatch(a, ctx) for a in n.args]
        A, b = fls.evaluate(dict(zip(names, vals)))
        systems.append(A if hasattr(A, "todense") else jnp.asarray(A))
        rhs.append(jnp.asarray(b).reshape(-1))
    solve, A0, B = members[0]._solve_fn, systems[0], jnp.stack(rhs, axis=1)
    X = None
    spec = getattr(members[0], "_linear_spec", None)
    # the solver context only where it decides something: building it assembles the form once more
    ctx0 = members[0]._fem_ref._linear_solver_context() if getattr(spec, "name", None) == "lu-cudss" else None
    if ctx0 is not None and (ctx0.get("complex_n") is not None or ctx0.get("symmetric")):
        # in-program LDL^T for a complex form or a provably symmetric real one -- the same choice a single solve
        # makes (jno.solve.lu): the block in ONE call, one factorisation
        from .linear import cudss_ldl_solve

        X = cudss_ldl_solve(A0, B, ctx0)
    if X is None and getattr(spec, "name", None) in ("lu-pardiso", "lu-cudss"):
        # PARDISO and the nvmath cuDSS LU take a block of right-hand sides in one call: one factorisation
        # (their adjoint is the same block call with the cotangents)
        from .linear import cudss_lu_solve, pardiso_lu_solve

        X = {"lu-pardiso": pardiso_lu_solve, "lu-cudss": cudss_lu_solve}[spec.name](A0, B)
    if X is None:
        # any other solver: column by column on the SAME matrix. Not `jax.vmap` over the solve: JAX's sparse
        # matvec (bcoo_dot_general) inside custom_linear_solve gets its cotangent shape wrong under vmap + grad
        # with an unbatched matrix (measured: "expected cotangent complex128[1, n], got [2, n]"). Host SuperLU
        # still factors once (it caches its factorisation by the matrix's content); a solver without a cache
        # factors per column, as separate solves would -- correct, without the saving.
        X = jnp.stack([solve(A0, B[:, i]) for i in range(B.shape[1])], axis=1)
    for i, n in enumerate(members):
        x = residual_gate(lambda v, _A=systems[i]: _A @ v, rhs[i], X[:, i], "shared factorisation", rtol=1e-6)
        rc = getattr(n, "_solve_recombine", None)
        memo[id(n)] = (n, rc(x) if rc is not None else x)
    return True
