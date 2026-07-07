"""Heterogeneous domain-decomposition coupling for ``jno.core([...])`` / ``jno.dd.couple([...])``.

Each subdomain problem (a ``jno.fdm([...])`` / ``jno.fem([...])``) owns a named region of one shared mesh
(``domain.region(name, poly)``). The driver infers the interface **geometrically** from the regions and
couples the subdomain solves — no ``on=`` argument, no hand-written interface equation. Two shapes:

* **Non-overlapping (a single interface line)** — the natural case when the domain tags *partition* the
  mesh and meet at a line. Coupled by a **Dirichlet-Neumann** iteration: the Dirichlet side takes the
  interface values, the Neumann side takes the interface flux. Value *and* flux continuity are enforced.
* **Overlapping (a 2-D strip)** — the subdomains share a band. Coupled by **overlapping Schwarz**: each
  side pins its complement to the neighbour's field (Dirichlet exchange); flux continuity emerges.

The mode is detected from the regions' intersection (area > 0 → overlap, else line). See
``plans/heterogeneous-domain-decomposition.md``.

The **overlap** coupling runs in JAX (the subdomain solves and the Schwarz exchange), so the combined
field is a JAX array; when a subdomain carries a trainable ``jno.np.parameter`` the coupled solve is a
differentiable trace node whose gradient reaches the parameter through ``jax.lax.custom_root`` (implicit
differentiation of the converged fixed point — no unrolled sweeps), giving differentiable **inverse**
domain decomposition. The geometry/partition bookkeeping (region masks, interface lines, edge lengths)
stays host-side ``numpy`` — it runs once and is never differentiated. The **line** Dirichlet-Neumann
mode is still a host-side forward solve.
"""

from __future__ import annotations

import numpy as np


def _region_mask(pts, geom):
    """Boolean mask of points ``pts`` inside a shapely region ``geom`` (used for nodes and element centroids)."""
    from shapely.geometry import Point

    g = geom.buffer(1e-9)
    try:
        import shapely  # vectorized (shapely >= 2.0.2)

        return np.asarray(shapely.contains_xy(g, np.asarray(pts)[:, 0], np.asarray(pts)[:, 1]))
    except (ImportError, AttributeError):
        return np.array([g.contains(Point(float(q[0]), float(q[1]))) for q in np.asarray(pts)])


def _element_partition(pts, tris, geom0):
    """Assign each element to region 0 (centroid in ``geom0``) or region 1, and return
    ``([nodes0, nodes1], gamma)`` where ``gamma`` = the interface nodes shared by both element sets."""
    cent = pts[tris].mean(1)
    in0 = _region_mask(cent, geom0)
    nodes0, nodes1 = np.unique(tris[in0]), np.unique(tris[~in0])
    gamma = np.intersect1d(nodes0, nodes1)
    return [nodes0, nodes1], gamma


def _interface_edge_lengths(pts, tris, gamma):
    """Nodal edge-lengths ``ell_i`` on the interface (half-sum of the interface edges at node ``i``) — the
    weight that turns a pointwise flux into a consistent FEM Neumann nodal load ``g_i * ell_i``."""
    gs = {int(x) for x in gamma}
    ell = np.zeros(len(pts))
    seen: set = set()
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            a, b = int(a), int(b)
            if a in gs and b in gs:
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                L = float(np.linalg.norm(pts[a] - pts[b]))
                ell[a] += L / 2
                ell[b] += L / 2
    return ell


def _interface_normal(pts, gamma, into_geom):
    """Unit normal of the (straight) interface line, oriented to point INTO ``into_geom`` (the Neumann
    region). This is the Dirichlet region's outward normal — the direction of the flux it exports."""
    from shapely.geometry import Point

    P = pts[gamma]
    c = P.mean(0)
    if len(gamma) >= 2:
        _, _, Vt = np.linalg.svd(P - c)
        d = Vt[0]  # dominant direction of the interface points
        nrm = np.array([-d[1], d[0]])
    else:
        nrm = np.array([1.0, 0.0])
    nrm = nrm / (np.linalg.norm(nrm) + 1e-30)
    if not into_geom.buffer(1e-9).contains(Point(float(c[0] + 1e-3 * nrm[0]), float(c[1] + 1e-3 * nrm[1]))):
        nrm = -nrm
    return nrm


def _is_fem(prob):
    """A jno.fem subdomain exposes its assembled (region-local) matrix/vector; a jno.fdm one does not."""
    return hasattr(prob, "A") and hasattr(prob, "b")


def _iter_nodes(root):
    """DFS over a trace node and all its children — views (``_expr``), binary/unary ops (``left``/``right``/
    ``operand``), function/operation nodes (``args``), derivative nodes (``variables`` — where a ``Jacobian``
    stashes the differentiation Variable, incl. a normal), and ``.bind`` coordinates (``_coord_vars``)."""
    seen: set = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        yield n
        for attr in ("_expr", "left", "right", "operand"):
            c = getattr(n, attr, None)
            if c is not None:
                stack.append(c)
        for attr in ("args", "variables"):
            seq = getattr(n, attr, None)
            if seq:
                stack.extend(c for c in seq if c is not None)
        cv = getattr(n, "_coord_vars", None)
        if isinstance(cv, dict):
            stack.extend(cv.values())


def _references_normal(node):
    """True if an interface condition contains a NORMAL derivative (references an ``n_interface_*`` tag) —
    a **flux** condition ``k*uA.d(n)-...``; a value condition ``uA(iface)-uB(iface)`` does not."""
    return any(isinstance(getattr(n, "tag", None), str) and n.tag.startswith("n_interface_") for n in _iter_nodes(node))


def _classify_interfaces(interface_conditions):
    """Split the declared interface conditions (each referencing an ``interface_*`` tag) into **flux**
    (carries a normal derivative ``.d(n)``) vs **value**; the split drives which exchange the driver uses."""
    conds = list(interface_conditions or [])
    flux = sum(1 for c in conds if _references_normal(c))
    return {"count": len(conds), "flux": flux, "value": len(conds) - flux}


# ---------------------------------------------------------------------------
# differentiable overlapping Schwarz (JAX)
#
# The forward coupling runs in JAX (subdomain solves + array ops), so the combined field is a JAX
# array. When a subdomain carries a trainable ``jno.np.parameter`` (an inverse parameter — e.g. a
# conductivity ``k`` in a FEM weak form) the coupled solve is returned as a differentiable trace node
# and the gradient reaches the parameter through ``jax.lax.custom_root``: the implicit function theorem
# differentiates the *converged* Schwarz fixed point without unrolling the sweeps (the truncated-sweep
# gradient would be wrong for Schwarz's slow convergence). This mirrors ``fem.solve()`` / ``fdm.solve()``
# — an array for a plain forward solve, a node for a parametric one — so ``jno.core([(couple([...]).
# solve() - u_obs).mse])`` recovers a parameter *through* the coupling (differentiable inverse DD).
# ---------------------------------------------------------------------------


def _fem_param_specs(prob):
    """``(names, nodes)`` of a FEM subdomain's runtime (trainable) parameters — empty for a
    non-parametric FEM. The names key ``FemLinearSystem.evaluate`` (re-assembling ``A(θ), b(θ)`` in the
    autodiff graph); the nodes are the trace ``Placeholder``s that become the coupled node's args."""
    from .trace import FemLinearSystem

    op = prob._op
    if isinstance(op, FemLinearSystem) and op.is_parametric:
        names = list(op.runtime_parameter_exprs)
        return names, [op.runtime_parameter_exprs[nm] for nm in names]
    return [], []


def _fdm_param_specs(prob):
    """``(lids, nodes, modules)`` of an FDM subdomain's trainable parameters — empty for a
    non-parametric FDM. ``lids``/``modules`` feed ``_steady_solve(extra_params=...)`` (the same value
    injection ``fdm.solve()`` uses for an inverse parameter)."""
    trainable = prob._trainable_params() if hasattr(prob, "_trainable_params") else {}
    lids = list(trainable)
    return lids, [trainable[lid] for lid in lids], {lid: trainable[lid].model.module for lid in lids}


def _subdomain_param_nodes(prob):
    """The trace nodes of a subdomain's trainable parameters (FEM or FDM) — the coupled node's args."""
    return (_fem_param_specs(prob) if _is_fem(prob) else _fdm_param_specs(prob))[1]


def _make_pinned_solver(prob, pin_idx, param_values):
    """A differentiable ``values -> full nodal field`` pinned solve for one subdomain, closing over that
    subdomain's resolved parameter values ``param_values`` (the crux-current arrays, in the order of
    :func:`_subdomain_param_nodes`).

    * **FEM**: re-assemble ``A(θ), b(θ)`` (``FemLinearSystem.evaluate`` — the same re-assembly
      ``fem.solve()`` differentiates through), row-pin the interface nodes and solve the dense system
      (``jnp.linalg.solve``; the DD meshes are small and the pinned stiffness is dense-direct's job — a
      matrix-free Krylov breaks down on it). Differentiable in ``θ`` and the pinned ``values``.
    * **FDM**: ``_steady_solve(extra_params=θ, extra_pins=(pin, values))`` — the same Newton-Krylov +
      ``custom_root`` solve ``jno.fdm`` uses, differentiable in ``θ`` and ``values``."""
    import jax.numpy as jnp

    from .trace import FemLinearSystem

    pin = jnp.asarray(pin_idx)

    if _is_fem(prob):
        op = prob._op
        names, _ = _fem_param_specs(prob)
        theta = {nm: v for nm, v in zip(names, param_values)}

        def solve(values):
            if isinstance(op, FemLinearSystem):
                A, b = op.evaluate(theta or None)
            else:
                A, b = op  # non-parametric (A_bcoo, b)
            A = jnp.asarray(A.todense()) if hasattr(A, "todense") else jnp.asarray(A)
            b = jnp.asarray(b).reshape(-1)
            A = A.at[pin].set(0.0).at[pin, pin].set(1.0)  # row-replacement: pinned rows -> identity
            rhs = b.at[pin].set(jnp.asarray(values))
            return jnp.linalg.solve(A, rhs)

        return solve

    # FDM
    import equinox as eqx

    lids, _, modules = _fdm_param_specs(prob)
    extra = {
        lid: eqx.tree_at(lambda m: m.value, modules[lid], jnp.asarray(v).astype(modules[lid].value.dtype))
        for lid, v in zip(lids, param_values)
    }

    def solve(values):
        return jnp.asarray(prob._steady_solve(extra_params=extra or None, extra_pins=(pin, jnp.asarray(values)))).reshape(
            -1
        )

    return solve


def _schwarz_multiplicative(solve0, solve1, mask0, c0, c1, U):
    """One multiplicative Schwarz sweep of the combined field ``U``: subdomain 0 solves with its
    complement ``c0`` pinned to ``U``, subdomain 1 then solves with its complement ``c1`` pinned to the
    *fresh* subdomain-0 field; combine on ``mask0`` (owned by 0, incl. the overlap). Returns ``(u0, u1,
    U_new)``."""
    import jax.numpy as jnp

    u0 = solve0(U[c0])
    u1 = solve1(u0[c1])
    return u0, u1, jnp.where(mask0, u0, u1)


def _schwarz_custom_root(solve0, solve1, mask0, c0, c1, n, *, tol, max_iter):
    """Converge the overlapping-Schwarz combined field and return it, **differentiable** via
    ``jax.lax.custom_root``. Forward: multiplicative sweeps (fast). Residual for the implicit diff: the
    additive map ``f0(U) = U - where(mask0, solve0(U[c0]), solve1(U[c1]))`` — it shares the same fixed
    point (so ``f0(U*) = 0``) and its Jacobian ``(I - G')`` is well-conditioned (Schwarz is a
    contraction), so the tangent solve is a matrix-free BiCGStab."""
    import jax
    import jax.numpy as jnp

    from .utils.solver.newton_krylov import bicgstab

    mask0, c0, c1 = jnp.asarray(mask0), jnp.asarray(c0), jnp.asarray(c1)

    def f0(U):
        return U - jnp.where(mask0, solve0(U[c0]), solve1(U[c1]))

    def forward(_f, U0):
        def cond(s):
            _U, r, k = s
            return (r > tol) & (k < max_iter)

        def body(s):
            U, _r, k = s
            _u0, _u1, Un = _schwarz_multiplicative(solve0, solve1, mask0, c0, c1, U)
            return Un, jnp.max(jnp.abs(Un - U)), k + 1

        U, _r, _k = jax.lax.while_loop(cond, body, (U0, jnp.asarray(jnp.inf), 0))
        return U

    bicg = lambda mv, rr: bicgstab(mv, rr, tol=1e-11, maxit=10000)
    tangent = lambda g, y: jax.lax.custom_linear_solve(g, y, bicg, transpose_solve=bicg)
    return jax.lax.custom_root(f0, jnp.zeros(n), forward, tangent)


def _schwarz_forward_eager(solve0, solve1, mask0, c0, c1, overlap, n, *, tol, max_iter):
    """Host-controlled multiplicative Schwarz for the plain (no trainable parameter) forward solve —
    the JAX counterpart of the old numpy loop: JAX subdomain solves + array ops, a Python convergence
    check (the iteration control is not differentiated). Returns ``(U, iters, overlap_jump)``."""
    import jax.numpy as jnp

    mask0j, c0j, c1j, ovj = jnp.asarray(mask0), jnp.asarray(c0), jnp.asarray(c1), jnp.asarray(overlap)
    U = jnp.zeros(n)
    jump, it = float("inf"), 0
    for it in range(1, max_iter + 1):
        u0, u1, U = _schwarz_multiplicative(solve0, solve1, mask0j, c0j, c1j, U)
        jump = float(jnp.max(jnp.abs(u0[ovj] - u1[ovj]))) if overlap.any() else 0.0
        if jump < tol:
            break
    return U, it, jump


class _Coupled:
    """A coupled domain-decomposition problem: subdomains + their regions, solved by the inferred method."""

    def __init__(self, subdomains, interface_conditions=None):
        if len(subdomains) != 2:
            raise NotImplementedError(
                "jno.dd: only 2 subdomains are supported for now (the complement of one is the other); "
                "N-subdomain coupling pins each complement to the combined field of all the others."
            )
        self._subdomains = list(subdomains)
        # Interface conditions declared in the constraint list (value `uA(iface)-uB(iface)` / flux
        # `k*uA.d(n)-...`). Currently they DECLARE the coupling the line-DN already enforces (value +
        # flux continuity); recognising them makes the coupling authored, not just inferred.
        self._interfaces = _classify_interfaces(interface_conditions)

    def solve(self, *, tol: float = 1e-7, max_iter: int = 400, return_info: bool = False):
        """Solve the coupled problem; return the combined nodal field. The coupling method (line
        Dirichlet-Neumann vs overlapping Schwarz) is inferred from whether the regions overlap."""
        probs = [p for p, _ in self._subdomains]
        geoms = [g for _, g in self._subdomains]
        inter = geoms[0].intersection(geoms[1])
        if float(getattr(inter, "area", 0.0)) > 1e-12:
            return self._solve_overlap(probs, geoms, tol=tol, max_iter=max_iter, return_info=return_info)
        return self._solve_line(probs, geoms, tol=tol, max_iter=max_iter, return_info=return_info)

    # -- non-overlapping: a single interface line, Dirichlet-Neumann -------------------------------
    def _solve_line(self, probs, geoms, *, tol, max_iter, theta=0.5, return_info=False):
        import scipy.linalg as sla

        # Line Dirichlet-Neumann is a host-side forward solve (no custom_root node). Fail LOUD on a
        # trainable parameter rather than silently returning a numpy array with the parameter baked in
        # at its initial value (which would never train through jno.core). Overlap → differentiable.
        if any(_subdomain_param_nodes(p) for p in probs):
            raise NotImplementedError(
                "jno.dd line (Dirichlet-Neumann) coupling is forward-only: a trainable jno.np.parameter "
                "in a non-overlapping subdomain is not differentiable yet — the coupled solve returns a "
                "plain array, so the parameter would silently never train. Give the subdomains an OVERLAP "
                "(the parametric coupled solve is then a differentiable node), or keep trainable "
                "parameters out of a line-coupled problem."
            )
        dom = probs[0].domain
        dim = int(getattr(dom, "dimension", 2))
        pts = np.asarray(dom.mesh_connectivity["points"])[:, :dim]
        tris = np.asarray(dom.mesh_connectivity["triangles"]).astype(int)
        n = pts.shape[0]
        region_nodes, gamma = _element_partition(pts, tris, geoms[0])

        fem_flags = [_is_fem(p) for p in probs]
        if not any(fem_flags):
            raise NotImplementedError(
                "jno.core line coupling needs at least one jno.fem subdomain (the Neumann side that "
                "consumes the interface flux). Two jno.fdm subdomains sharing a line would need an FDM "
                "Neumann flux condition (not in v1) — give them an overlap for value-exchange instead."
            )
        ni = fem_flags.index(True)  # Neumann side = a FEM subdomain
        di = 1 - ni  # Dirichlet side = the other subdomain
        fem, other = probs[ni], probs[di]
        Nnodes, Dnodes = region_nodes[ni], region_nodes[di]
        Nint = np.setdiff1d(Nnodes, gamma)  # Neumann-region interior (fed to the Dirichlet solve)
        nonN = np.setdiff1d(np.arange(n), Nnodes)  # empty rows of the region-local FEM matrix
        ell = _interface_edge_lengths(pts, tris, gamma)
        nrm = _interface_normal(pts, gamma, geoms[ni])  # Dirichlet outward normal = into Neumann region

        # Neumann (FEM) solver: pin the empty non-region rows; Gamma is a FREE DOF carrying the flux load.
        AN = np.asarray(fem.A).copy()
        bN = np.asarray(fem.b).reshape(-1)
        AN[nonN, :] = 0.0
        AN[nonN, nonN] = 1.0
        luN = sla.lu_factor(AN)

        # Dirichlet-side solver (pin Gamma to lambda + the Neumann interior to its current field), built once.
        if _is_fem(other):
            AD = np.asarray(other.A).copy()
            bD = np.asarray(other.b).reshape(-1)
            nonD = np.setdiff1d(np.arange(n), Dnodes)
            pinD = np.union1d(nonD, gamma).astype(int)
            AD2 = AD.copy()
            AD2[pinD, :] = 0.0
            AD2[pinD, pinD] = 1.0
            luD = sla.lu_factor(AD2)

            def dirichlet_solve(lam, uN):
                rhs = bD.copy()
                rhs[nonD] = 0.0
                rhs[gamma] = lam
                return sla.lu_solve(luD, rhs)

            def dirichlet_flux(uD):  # exact consistent reaction (same basis) -> nodal flux out of D
                return (AD @ uD - bD)[gamma]
        else:
            from .fdm import gradient as _grad

            dsolve = other.pinned_solver(np.concatenate([gamma, Nint]).astype(int))

            def dirichlet_solve(lam, uN):
                return np.asarray(dsolve(np.concatenate([np.asarray(lam), uN[Nint]]))).reshape(-1)

            def dirichlet_flux(uD):  # pointwise strong-form flux -> consistent nodal load via edge-length
                g = np.asarray(_grad(uD, dom))
                return (g[gamma] @ nrm) * ell[gamma]

        lam = np.zeros(len(gamma))
        uN = np.zeros(n)
        uD = np.zeros(n)
        step = np.inf
        it = 0
        for it in range(1, max_iter + 1):
            uD = dirichlet_solve(lam, uN)
            flux = dirichlet_flux(uD)
            rhs = bN.copy()
            rhs[nonN] = 0.0
            rhs[gamma] = rhs[gamma] - flux  # Neumann load = -(flux out of the Dirichlet region)
            uN = sla.lu_solve(luN, rhs)
            new = (1 - theta) * lam + theta * uN[gamma]
            step = float(np.max(np.abs(new - lam))) if len(gamma) else 0.0
            lam = new
            if step < tol:
                break

        own_N = np.isin(np.arange(n), Nnodes) & ~np.isin(np.arange(n), gamma)
        combined = np.where(own_N, uN, uD)
        combined[gamma] = lam
        if return_info:
            return combined, {
                "iterations": it,
                "interface_step": step,
                "gamma_nodes": int(len(gamma)),
                "mode": "line-DN",
                "interfaces": self._interfaces,
            }
        return combined

    # -- overlapping: a 2-D strip, overlapping Schwarz (value exchange) ----------------------------
    def _solve_overlap(self, probs, geoms, *, tol, max_iter, return_info=False):
        # A region-tagged FEM (needed so `jno.core` can detect it) assembles region-local (RegionMask),
        # which can't reconcile an overlap band — its artificial boundary reaches no neighbour cells. Rebuild
        # any such subdomain WHOLE-MESH (one cheap re-assemble, reused across all iterations) so
        # complement-pinning closes the overlap; the region label is preserved for the masks.
        probs = [p._as_whole_mesh() if (_is_fem(p) and getattr(p, "region", None) is not None) else p for p in probs]
        dom = probs[0].domain
        dim = int(getattr(dom, "dimension", 2))
        pts = np.asarray(dom.mesh_connectivity["points"])[:, :dim]
        n = pts.shape[0]
        masks = [_region_mask(pts, g) for g in geoms]
        complements = [np.where(~m)[0].astype(int) for m in masks]
        overlap = masks[0] & masks[1]
        mask0 = masks[0]  # owned by subdomain 0 (includes the overlap)

        # Trainable parameters across the subdomains (a FEM coefficient / an FDM parameter). If any, the
        # coupled solve is a differentiable node (below); the values arrive in this order at solve time.
        counts = [len(_subdomain_param_nodes(p)) for p in probs]
        param_nodes = [nd for p in probs for nd in _subdomain_param_nodes(p)]

        def _solvers(values):
            """Build the two differentiable pinned solvers, handing each subdomain its own parameters."""
            solvers, off = [], 0
            for i, p in enumerate(probs):
                solvers.append(_make_pinned_solver(p, complements[i], values[off : off + counts[i]]))
                off += counts[i]
            return solvers

        if param_nodes:
            # Parametric: return a differentiable trace node (∂u/∂θ flows through custom_root, so
            # `jno.core([(couple([...]).solve() - u_obs).mse])` recovers θ *through* the coupling).
            from .trace import FunctionCall

            def _run(*values):
                s0, s1 = _solvers(list(values))
                return _schwarz_custom_root(s0, s1, mask0, complements[0], complements[1], n, tol=tol, max_iter=max_iter)

            node = FunctionCall(_run, param_nodes, name="dd_solve")
            node._domain = dom  # so jno.core infers the domain from the graph
            return node

        # Plain forward (no trainable parameter): host-controlled Schwarz over the JAX subdomain solves.
        # jit each solver so it compiles ONCE and is reused across sweeps (safe here — no parameter
        # tracers are closed over; the parametric branch above must NOT jit, or the θ-gradient is lost).
        import jax

        s0, s1 = (jax.jit(s) for s in _solvers([]))
        combined, iters, jump = _schwarz_forward_eager(
            s0, s1, mask0, complements[0], complements[1], overlap, n, tol=tol, max_iter=max_iter
        )
        if return_info:
            return combined, {
                "iterations": iters,
                "overlap_jump": jump,
                "mode": "overlap-Schwarz",
                "interfaces": self._interfaces,
            }
        return combined


def couple(subdomains, interface_conditions=None):
    """Couple subdomain problems by domain decomposition on a shared mesh.

    ``subdomains``: a list of ``(problem, region)`` pairs, where ``problem`` is a subdomain solve
    (``jno.fdm([...])`` / ``jno.fem([...])``) authored with its PDE + outer boundary conditions, and
    ``region`` is the shapely geometry it owns. ``interface_conditions``: optional residuals declaring the
    coupling in jNO syntax (value ``uA(iface)-uB(iface)`` / flux ``k*uA.d(n)-...`` on an ``interface_*``
    tag). The interface is inferred from the regions: a single line (partitioning tags) is coupled by
    Dirichlet-Neumann, an overlap by Schwarz.

    ``.solve()`` returns the combined nodal field as a JAX array — **or**, when a subdomain carries a
    trainable ``jno.np.parameter`` and the regions overlap, a differentiable trace node (exactly as
    ``fem.solve()`` / ``fdm.solve()`` do). The gradient reaches the parameter through the converged
    Schwarz fixed point (``jax.lax.custom_root``), so a differentiable **inverse** domain-decomposition
    problem is just::

        kL   = jno.np.parameter((1,), name="kL")        # coefficient to recover
        femA = jno.fem([kL * (ui.x*vi.x + ui.y*vi.y) - f*vi, ...])
        node = jno.dd.couple([(femA, boxA), (fdmB, boxB)]).solve()
        jno.core([(node - u_obs).mse]).solve(epochs)    # recovers kL THROUGH the coupling

    The user-facing surface for the plain forward coupling is ``jno.core([...])``, which builds this
    automatically."""
    return _Coupled(subdomains, interface_conditions)
