"""Geometry to PEEC elements — the bridge between :mod:`jno.geometry` and the integral kernel.

A partial-element method never meshes: a conductor becomes a chain of straight filaments carrying
its centreline and cross-section, and the operator is then the Neumann double integral, which is
:func:`jno.utils.solver.kernel.pair_quadratic` with ``mom = tangent × length``. So this module holds
discretisation and nothing else — no physics the kernel already has, no solver.

Everything returns arrays shaped for the kernel, so the caller writes::

    f = line_filaments(wire, size=0.5)
    L = float(pair_quadratic(f.pos, f.mom, lambda r: 1/r, f.self_g, f.group)) * MU0 / (4 * jnp.pi)

Filaments carry their **analytic** cross-section, which is the reason to prefer them over a meshed
conductor wherever the geometry allows. A meshed cylinder is an inscribed polygon: at seven or eight
points around a 375 µm bond wire the mesh keeps only ~88 % of the true area, which makes the wire
~14 % too resistive. A filament has no faceting to lose.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .kernel import wire_self

__all__ = ["Filaments", "line_filaments", "network_impedance"]


class Filaments(NamedTuple):
    """A conductor discretised into straight filaments, shaped for the kernel and for a circuit solve.

    ``pos``/``mom``/``self_g``/``group`` go straight to :func:`~jno.utils.solver.kernel.pair_quadratic`.
    ``incidence`` and ``length`` are what a network solve needs on top: which nodes each filament
    joins, and how long it is.
    """

    pos: object  # (N*quad, 3) sub-point positions
    mom: object  # (N*quad, 3) tangent * length * Gauss weight
    self_g: object  # (N,)      per-filament self term
    group: object  # (N*quad,) filament label for each sub-point
    incidence: object  # (n_node, N) +1 where a filament leaves a node, -1 where it enters
    length: object  # (N,)
    radius: object  # (N,)    filament radius
    nodes: object  # (n_node, 3) node positions, which is how a port is addressed


def _leaf(shape, what):
    node = getattr(shape, "_node", None)
    prim = node[1] if (isinstance(node, tuple) and node and node[0] == "leaf") else None
    if type(prim).__name__ != what:
        got = "no plan at all" if node is None else f"a {node[0] if prim is None else type(prim).__name__!r} plan"
        raise NotImplementedError(
            f"peec: expected a single {what} primitive, got {got}. A filament discretisation needs a "
            "centreline and a cross-section; a CSG plan has neither. Build the conductor from "
            "Shape.line (a tube along a polyline), or discretise a solid as a lattice instead."
        )
    return prim


def line_filaments(shape, size: float = None, quad: int = 3):
    """Discretise :meth:`jno.Shape.line` conductors into filaments, with Gauss sub-points.

    Args:
        shape: a ``Shape`` whose plan is a single ``Line`` leaf, or a sequence of them. A sequence
            becomes ONE network: lines that share an endpoint share a node, which is how a branch,
            a tee or a parallel pair is expressed -- there is no separate "join" step.
        size: target filament length. Defaults to each shape's own ``size=``.
        quad: Gauss points per filament. One point is 7.8 % low against a closed form on the worst
            case (collinear neighbours); 2 gives 2.5 %, 3 gives 1.2 %, 8 gives 0.21 %. Three is the
            default because it is where the curve flattens against its cost.

    Returns:
        :class:`Filaments`.

    Nodes are the filament endpoints, deduplicated onto a grid of ``1e-9 x`` the shortest filament.
    A CLOSED polyline therefore yields a loop rather than a chain -- which is what makes a current
    path possible at all -- and two lines meeting at a point are electrically joined. The grid is
    needed rather than exact equality because a shared vertex reached from its two segments differs
    in the last bits; it is nine orders below the element size, so it cannot fuse two terminals that
    are meant to stay apart.

    Each polyline is subdivided so that no filament exceeds ``size``, and each original vertex stays
    a filament boundary — a bend must not fall inside a straight element.
    """
    shapes = list(shape) if isinstance(shape, (list, tuple)) else [shape]
    starts, tangs, lens, ends, radii = [], [], [], [], []
    for sh in shapes:
        prim = _leaf(sh, "Line")
        h = float(size if size is not None else (sh._size if sh._size is not None else 0.0))
        if h <= 0:
            raise ValueError(
                "peec.line_filaments: no filament length. Pass size=, or give the Shape a size= when "
                "you build it — a filament count cannot be guessed from the geometry alone."
            )
        P = np.asarray(prim.points, dtype=float).reshape(-1, 3)
        for a, d in zip(P[:-1], P[1:] - P[:-1]):
            ln0 = float(np.linalg.norm(d))
            if ln0 <= 0.0:
                continue
            k = max(1, int(np.ceil(ln0 / h)))  # vertices stay filament boundaries: subdivide within a segment
            u = d / ln0
            step = ln0 / k
            for j in range(k):
                starts.append(a + u * (step * (j + 0.5)))
                tangs.append(u)
                lens.append(step)
                ends.append((a + u * (step * j), a + u * (step * (j + 1))))
                radii.append(prim.r)
    if not starts:
        raise ValueError("peec.line_filaments: the polyline has no segment longer than zero.")

    cen = np.asarray(starts)
    tan = np.asarray(tangs)
    ln = np.asarray(lens)
    rad = np.asarray(radii)
    n = len(ln)

    gx, gw = np.polynomial.legendre.leggauss(int(quad))
    # sub-points along each filament, and moments that sum to `tangent * length` per filament
    pos = (cen[:, None, :] + 0.5 * ln[:, None, None] * gx[None, :, None] * tan[:, None, :]).reshape(-1, 3)
    mom = (tan[:, None, :] * (ln[:, None] * gw[None, :] * 0.5)[:, :, None]).reshape(-1, 3)
    group = np.repeat(np.arange(n), int(quad))
    self_g = np.asarray(wire_self(jnp.asarray(ln), jnp.asarray(rad)))

    # nodes = filament endpoints snapped to a grid far below the element size
    tol = 1e-9 * float(ln.min())
    key, rows, xyz = {}, [], []
    for k, (a, b) in enumerate(ends):
        for pt, sign in ((a, +1.0), (b, -1.0)):
            t = tuple(np.round(np.asarray(pt) / tol).astype(np.int64).tolist())
            if t not in key:
                key[t] = len(rows)
                rows.append(np.zeros(n))
                xyz.append(np.asarray(pt, dtype=float))
            rows[key[t]][k] += sign
    return Filaments(
        jnp.asarray(pos),
        jnp.asarray(mom),
        jnp.asarray(self_g),
        group,
        jnp.asarray(np.asarray(rows)),
        jnp.asarray(ln),
        jnp.asarray(rad),
        jnp.asarray(np.asarray(xyz)),
    )


def network_impedance(fil: Filaments, sigma, port, omega: float = 0.0, mu0: float = 4e-7 * np.pi):
    """Terminal impedance of a filament network, from the PEEC circuit equations.

    The unknowns are the filament currents ``I`` and the node potentials ``phi``; the equations are
    Ohm plus the partial inductances along each filament, and Kirchhoff's current law at each node::

        Z I - A' phi = 0        Z = diag(R) + j w Lp,  A = incidence
        (A I)_n      = 0        at every node that is not a terminal

    which is Ruehli's formulation (Ruehli, *Equivalent circuit models for three-dimensional
    multiconductor systems*, IEEE Trans. MTT 22(3), 1974, §III). One volt is impressed across the
    two terminals, so the answer is ``1 / I_terminal``.

    Args:
        fil: the network, from :func:`line_filaments`.
        sigma: conductivity, scalar or per filament.
        port: ``(point_a, point_b)`` — the two terminals, given as coordinates and matched to the
            nearest node. A closed loop has no terminal: open it where the source sits.
        omega: angular frequency. ``0`` gives the DC resistance and a real answer.
        mu0: permeability of the surrounding medium.

    Returns:
        ``(Z, I)`` — the terminal impedance and the per-filament currents at 1 V.

    This is the SMALL-network path: it forms the dense ``Lp`` and factors the ``(N+n) x (N+n)``
    system, so it is O(N^2) in memory and O(N^3) in time. It also carries PEEC's own limitation --
    each filament has ONE current, so a conductor wide against the skin depth needs to be split
    across its section, by the caller, into filaments that can carry different currents.
    """
    from .kernel import pair_matrix

    A = jnp.asarray(fil.incidence)
    nn, ne = A.shape
    sig = jnp.broadcast_to(jnp.asarray(sigma, dtype=float), (ne,))
    R = jnp.asarray(fil.length) / (sig * jnp.pi * jnp.asarray(fil.radius) ** 2)
    Lp = pair_matrix(fil.pos, fil.mom, lambda r: 1.0 / r, fil.self_g, group=fil.group) * (mu0 / (4.0 * jnp.pi))
    Z = jnp.diag(R.astype(complex)) + 1j * omega * Lp

    nodes = np.asarray(fil.nodes)
    ia, ib = (int(np.argmin(((nodes - np.asarray(p, dtype=float)) ** 2).sum(1))) for p in port)
    if ia == ib:
        raise ValueError(
            "peec.network_impedance: both terminals landed on the same node, so there is no port. "
            "A closed loop must be opened where its source sits — leave a gap in the polyline."
        )

    # KCL everywhere except the two terminals, whose rows instead impress the potential
    keep = jnp.asarray(np.setdiff1d(np.arange(nn), [ia, ib]))
    Ak = A[keep]
    P = jnp.zeros((2, nn), dtype=complex).at[0, ia].set(1.0).at[1, ib].set(1.0)
    top = jnp.concatenate([Z, -A.T.astype(complex)], axis=1)
    mid = jnp.concatenate([Ak.astype(complex), jnp.zeros((Ak.shape[0], nn), dtype=complex)], axis=1)
    bot = jnp.concatenate([jnp.zeros((2, ne), dtype=complex), P], axis=1)
    M = jnp.concatenate([top, mid, bot], axis=0)
    rhs = jnp.concatenate([jnp.zeros(ne + Ak.shape[0], dtype=complex), jnp.array([1.0 + 0j, 0.0 + 0j])])

    x = jnp.linalg.solve(M, rhs)
    cur = x[:ne]
    return 1.0 / (A[ia] @ cur), cur
