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


def _classify_interfaces(interface_conditions):
    """Summarise the interface conditions declared in the constraint list (each recognised by referencing
    an ``interface_*`` tag). ``count`` is the number of declared conditions. Splitting them into flux vs
    value needs the interface normal to survive view arithmetic (``uA.d(n) - uB.d(n)`` currently drops it
    from the walkable trace) — deferred to the material-``k`` flux-weighting step."""
    return {"count": len(list(interface_conditions or []))}


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
        dom = probs[0].domain
        dim = int(getattr(dom, "dimension", 2))
        pts = np.asarray(dom.mesh_connectivity["points"])[:, :dim]
        n = pts.shape[0]
        masks = [_region_mask(pts, g) for g in geoms]
        complements = [np.where(~m)[0].astype(int) for m in masks]
        overlap = masks[0] & masks[1]
        sols = [np.zeros(n), np.zeros(n)]
        solvers = [probs[i].pinned_solver(complements[i]) for i in range(2)]

        jump = np.inf
        iters = 0
        for iters in range(1, max_iter + 1):
            for i in range(2):
                neighbour = sols[1 - i]
                sols[i] = np.asarray(solvers[i](neighbour[complements[i]])).reshape(-1)
            jump = float(np.max(np.abs(sols[0][overlap] - sols[1][overlap]))) if overlap.any() else 0.0
            if jump < tol:
                break

        combined = np.where(masks[0], sols[0], sols[1])
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
    Dirichlet-Neumann, an overlap by Schwarz. ``.solve()`` returns the combined nodal field. The
    user-facing surface is ``jno.core([...])``, which builds this automatically."""
    return _Coupled(subdomains, interface_conditions)
