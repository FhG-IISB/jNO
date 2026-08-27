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

from .kernel import bar_self, wire_self

__all__ = [
    "Filaments",
    "line_filaments",
    "bar_filaments",
    "network_impedance",
    "port_spec",
    "terminal_nodes",
    "solve_network",
]


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
    area: object  # (N,)    conducting cross-section, which is all the solve needs of the shape
    nodes: object  # (n_node, 3) node positions, which is how a port is addressed
    part: object  # (N,)    index of the shape each filament came from, so per-conductor data can be spread
    lattice: object = None  # grid description when the elements sit on one, which is what the FFT path needs


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
    starts, tangs, lens, ends, radii, part = [], [], [], [], [], []
    for si, sh in enumerate(shapes):
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
                part.append(si)
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
    area = np.pi * rad**2

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
        jnp.asarray(area),
        jnp.asarray(np.asarray(xyz)),
        np.asarray(part, dtype=int),
        None,
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
    R = jnp.asarray(fil.length) / (sig * jnp.asarray(fil.area))
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


# ----------------------------------------------------------------------------------------------
# Reading a constraint list into ports
# ----------------------------------------------------------------------------------------------


def _trial_names(expr, out=None):
    """Every trial-function name appearing in ``expr`` — which field a constraint is written on."""
    from jno.trace import TrialFunction

    out = set() if out is None else out
    inner = object.__getattribute__(expr, "_expr") if hasattr(type(expr), "_expr") else getattr(expr, "_expr", None)
    if inner is not None and inner is not expr:  # a bound view wraps the node it was built from
        _trial_names(inner, out)
    if isinstance(expr, TrialFunction):
        out.add(str(getattr(expr, "name", "?")))
    for a in getattr(expr, "args", None) or []:
        _trial_names(a, out)
    for side in ("left", "right"):
        sub = getattr(expr, side, None)
        if sub is not None:
            _trial_names(sub, out)
    return out


def port_spec(constraints, current="i", potential="v"):
    """Read a PEEC constraint list into ``(sources, currents, grounds)``.

    Three forms, and only three::

        v(A) - v(B) - g     a source of g volts from terminal A to terminal B
        v(A) - g            terminal A held at g volts (g = 0 is the ground / reference)
        i(A) - g            terminal A carries g amps (g = 0 is an open terminal)

    A relation across TWO regions only means something on the potential, so ``i(A) - i(B)`` is
    refused rather than guessed at.

    Pure: it reads names and constants off the trace and touches no geometry, so what a terminal
    NAME resolves to is the caller's business.
    """
    from jno._fem import _bare, _scalar_const
    from jno.trace.views import _tag_of_coord_vars

    sources, currents, grounds = [], [], []
    for c in constraints:
        names = _trial_names(c)
        if len(names) != 1:
            raise ValueError(
                f"jno.peec: a constraint must be written on exactly one of {potential!r} (a terminal "
                f"potential) or {current!r} (a terminal current); this one names {sorted(names) or 'neither'}. "
                "Split it into one constraint per field."
            )
        field = names.pop()
        if field not in (current, potential):
            raise ValueError(
                f"jno.peec: unknown field {field!r}. The symbols from peec_symbols() are "
                f"{current!r} (terminal current) and {potential!r} (terminal potential)."
            )
        bare = _bare(c)
        g = _scalar_const(bare.right) if getattr(bare, "op", None) == "-" else None
        tie = getattr(c, "_periodic_tie", None)
        if tie is not None:
            if field != potential:
                raise ValueError(
                    f"jno.peec: `{field}(A) - {field}(B)` is not a port. A relation between two terminals is a "
                    f"voltage source, so write it on {potential!r}: `{potential}(A) - {potential}(B) - volts`."
                )
            sources.append((tie[0], tie[1], complex(g if g is not None else 0.0)))
            continue
        cv = getattr(c, "_coord_vars", None)
        tag = _tag_of_coord_vars(cv) if cv else None
        if not isinstance(tag, str):
            raise ValueError(
                "jno.peec: a constraint must be bound to a named terminal — bind the region first, as in "
                "`p = e.variable('DC+', split=True)[:3]` and then `v(*p) - 0.0`."
            )
        (grounds if field == potential else currents).append((tag, complex(g if g is not None else 0.0)))
    return sources, currents, grounds


def terminal_nodes(fil: Filaments, where):
    """Indices of the network nodes lying inside ``where`` — a terminal is a REGION, not a point.

    ``where`` is anything with a ``.contains(points)`` (a :class:`jno.Shape`) or a callable taking
    ``(N, 3)`` and returning a boolean mask. A real pad has many filament ends on it, and they are
    held at one potential by :func:`solve_network`.
    """
    pts = np.asarray(fil.nodes)
    hit = where.contains(pts) if hasattr(where, "contains") else where(pts)
    idx = np.flatnonzero(np.asarray(hit).reshape(-1))
    if idx.size == 0:
        raise ValueError(
            "peec.terminal_nodes: no network node lies in that terminal. A terminal must contain at "
            "least one filament END — a region that only crosses a filament's middle has nothing to "
            "connect to; split the polyline so a vertex lands there."
        )
    return idx


def solve_network(fil: Filaments, sigma, terminals, sources, grounds=(), currents=(), omega=0.0, mu0=4e-7 * np.pi):
    """Solve the PEEC circuit for a network whose terminals are node SETS.

    The general form of :func:`network_impedance`: a terminal is a pad carrying many filament ends,
    held at one potential, and a problem may have several of them. The equations are Ruehli's
    (IEEE Trans. MTT 22(3), 1974, sec. III)::

        Z I - A' phi = 0                       Z = diag(R) + j w Lp,  A = incidence
        phi_n = phi_T      for n in T          a terminal is equipotential
        (A I)_n = 0                            at every node outside a terminal

    and one row per terminal, from the ports: a source fixes ``phi_A - phi_B``, a ground fixes
    ``phi_T``, a current fixes the net current injected there, and a terminal named by nothing at
    all is open -- which is what ``(A I)_T = 0`` already says, so writing ``i(T) - 0`` is a way to
    say out loud what omitting it would mean anyway.

    Args:
        fil: the network.
        sigma: conductivity, scalar or per filament.
        terminals: ``{name: node indices}``, from :func:`terminal_nodes`.
        sources/grounds/currents: as returned by :func:`port_spec`.
        omega: angular frequency, rad/s.
        mu0: permeability of the surrounding medium.

    Returns:
        ``(cur, phi, inj)`` — filament currents, node potentials, and ``{terminal: injected current}``.

    A source breaks current balance at BOTH its terminals, so each one needs a row: the source
    supplies the first, and the second needs a ground. With exactly one source and no ground given,
    its negative side is grounded -- a gauge choice that moves no current and changes no impedance.
    With more than one, the reference is ambiguous and is asked for rather than picked.
    """
    from .kernel import pair_matrix

    A = jnp.asarray(fil.incidence)
    nn, ne = A.shape
    sig = jnp.broadcast_to(jnp.asarray(sigma, dtype=float), (ne,))
    R = jnp.asarray(fil.length) / (sig * jnp.asarray(fil.area))
    Lp = pair_matrix(fil.pos, fil.mom, lambda r: 1.0 / r, fil.self_g, group=fil.group) * (mu0 / (4.0 * jnp.pi))
    Z = jnp.diag(R.astype(complex)) + 1j * float(omega) * Lp

    names = list(terminals)
    idx = {t: np.asarray(terminals[t], dtype=int) for t in names}
    for t, ids in idx.items():
        if ids.size == 0:
            raise ValueError(f"jno.peec: terminal {t!r} contains no network node.")
    owner = {}
    for t, ids in idx.items():
        for n in ids.tolist():
            if n in owner:
                raise ValueError(
                    f"jno.peec: node {n} is in both terminal {owner[n]!r} and {t!r}. Terminals must not "
                    "overlap -- two pads holding the same conductor end at different potentials has no "
                    "meaning."
                )
            owner[n] = t

    # one row per terminal; a source claims its positive side, everything else claims itself
    row = {}

    def claim(t, what):
        if t not in idx:
            raise ValueError(f"jno.peec: {what[0]} names terminal {t!r}, which is not a terminal here. Known: {names}.")
        if t in row:
            raise ValueError(
                f"jno.peec: terminal {t!r} is constrained twice ({row[t][0]} and {what[0]}). Each terminal "
                "takes one condition."
            )
        row[t] = what

    for a, b, g in sources:
        claim(a, ("a source", ("source", a, b, g)))
    for t, g in grounds:
        claim(t, ("a fixed potential", ("ground", t, g)))
    for t, g in currents:
        claim(t, ("a fixed current", ("current", t, g)))

    # a source breaks current balance at its negative side too, so that side needs a row of its own
    floating = [b for _a, b, _g in sources if b not in row]
    if floating:
        if len(sources) == 1:
            claim(floating[0], ("the implied reference", ("ground", floating[0], 0.0 + 0j)))
        else:
            raise ValueError(
                f"jno.peec: {len(sources)} sources and no reference for {floating}. With one source the "
                "negative side is grounded for you; with several the reference is ambiguous, so name it: "
                "add `v(REF) - 0.0`."
            )

    free = jnp.asarray(np.array([n for n in range(nn) if n not in owner], dtype=int))
    rows, rhs = [], []
    rows.append(jnp.concatenate([Z, -A.T.astype(complex)], axis=1))
    rhs.append(jnp.zeros(ne, dtype=complex))
    if free.size:  # current balance away from the terminals
        rows.append(jnp.concatenate([A[free].astype(complex), jnp.zeros((free.size, nn), dtype=complex)], axis=1))
        rhs.append(jnp.zeros(free.size, dtype=complex))
    for t in names:  # a terminal is equipotential: tie its nodes to its first
        ids = idx[t]
        if ids.size > 1:
            tie = jnp.zeros((ids.size - 1, ne + nn), dtype=complex)
            for k, n in enumerate(ids[1:].tolist()):
                tie = tie.at[k, ne + n].set(1.0).at[k, ne + int(ids[0])].set(-1.0)
            rows.append(tie)
            rhs.append(jnp.zeros(ids.size - 1, dtype=complex))
    for t in names:  # and one row for the terminal itself
        r = jnp.zeros(ne + nn, dtype=complex)
        kind, *rest = row[t][1] if t in row else ("open", t, 0.0 + 0j)
        if kind == "source":
            a, b, g = rest
            r = r.at[ne + int(idx[a][0])].set(1.0).at[ne + int(idx[b][0])].set(-1.0)
            rhs.append(jnp.asarray(g, dtype=complex))
        elif kind == "ground":
            _t, g = rest
            r = r.at[ne + int(idx[t][0])].set(1.0)
            rhs.append(jnp.asarray(g, dtype=complex))
        else:  # a fixed injected current, or an open terminal (which is zero injected current)
            _t, g = rest
            r = r.at[:ne].set(A[idx[t]].sum(0).astype(complex))
            rhs.append(jnp.asarray(g, dtype=complex))
        rows.append(r[None, :])
    M = jnp.concatenate(rows, axis=0)
    b = jnp.concatenate([jnp.atleast_1d(x) for x in rhs])
    if M.shape[0] != M.shape[1]:
        raise ValueError(f"jno.peec: built a {M.shape[0]}x{M.shape[1]} system; this is a bug, please report it.")

    _refuse_disconnected(np.asarray(A), idx, sources, grounds, currents)
    x = jnp.linalg.solve(M, b)
    if not bool(jnp.all(jnp.isfinite(x))):
        raise ValueError(
            "jno.peec: the circuit equations are singular — the solve returned non-finite currents. "
            "The usual cause is a part of the network that no port reaches, or a conductor with no "
            "path back to the reference."
        )
    cur, phi = x[:ne], x[ne:]
    inj = {t: A[idx[t]].sum(0) @ cur for t in names}
    return cur, phi, inj


def _refuse_disconnected(A, idx, sources, grounds, currents):
    """A port pair with no metal between them is a modelling error, not an infinite impedance.

    ``jnp.linalg.solve`` on a singular system returns ``inf``/``nan`` without complaining, and an
    infinite resistance reads like a physical answer. Two conductors that were meant to touch but do
    not is the common way to get here, so the check names the terminals rather than the matrix.
    """
    nn, ne = A.shape
    root = np.arange(nn)

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    for k in range(ne):  # a filament joins the nodes it touches
        touch = np.flatnonzero(A[:, k] != 0.0)
        for n in touch[1:]:
            ra, rb = find(int(touch[0])), find(int(n))
            if ra != rb:
                root[ra] = rb
    for ids in idx.values():  # a terminal is one electrical point
        for n in ids[1:]:
            ra, rb = find(int(ids[0])), find(int(n))
            if ra != rb:
                root[ra] = rb
    for a, b, _g in sources:
        if find(int(idx[a][0])) != find(int(idx[b][0])):
            raise ValueError(
                f"jno.peec: no conducting path between terminals {a!r} and {b!r}, so no current can "
                "flow and the impedance is not finite. Conductors are joined where the geometry says "
                "the metal touches — check that the parts meant to be in contact actually overlap."
            )


def bar_filaments(shape, size: float = None, quad: int = 3):
    """Discretise a box conductor into a regular lattice of rectangular bars.

    A solid does not have a centreline, so a line's discretisation does not apply. The volume is cut
    into a regular grid of cells; the NODES are cell centres and the ELEMENTS are the bars joining
    adjacent centres, one family per axis. That is the standard volume partial-element mesh (Ruehli,
    IBM J. Res. Dev. 16(5), 1972, sec. IV), and each bar takes its self term from
    :func:`~jno.utils.solver.kernel.bar_self`.

    Two properties of this lattice matter beyond expressing the solid:

    * a bar's current runs along ONE axis, and ``mom_i . mom_j`` vanishes between perpendicular
      bars, so the partial-inductance operator is block diagonal by direction;
    * within a direction the bars sit on a regular grid and the kernel depends only on separation,
      so each block is block-Toeplitz -- which is what
      :func:`~jno.utils.solver.kernel.lattice_operator` applies by FFT.

    Args:
        shape: a ``Shape`` whose plan is a single ``Box`` leaf.
        size: target cell pitch. Defaults to the shape's own ``size=``.
        quad: Gauss points along each bar.

    Returns:
        :class:`Filaments`, with ``lattice`` describing the grid the FFT path needs.

    The cell count per axis is ``max(1, round(extent / size))``, so the pitch is the extent divided
    by a whole number rather than ``size`` exactly -- a lattice has to close on the box.
    """
    prim = _leaf(shape, "Box")
    h = float(size if size is not None else (shape._size if shape._size is not None else 0.0))
    if h <= 0:
        raise ValueError(
            "peec.bar_filaments: no cell pitch. Pass size=, or give the Shape a size= when you build "
            "it — a lattice count cannot be guessed from the geometry alone."
        )
    lo = np.array([prim.x0, prim.y0, prim.z0], dtype=float)
    hi = np.array([prim.x1, prim.y1, prim.z1], dtype=float)
    ext = hi - lo
    n = np.maximum(1, np.round(ext / h).astype(int))
    d = ext / n  # cell pitch per axis, closing exactly on the box
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError(f"peec.bar_filaments: the box has a zero or non-finite extent {tuple(ext)}.")
    if int(n.max()) < 2:
        raise ValueError(
            f"peec.bar_filaments: a pitch of {h} puts one cell across every axis of this box, so no bar "
            "joins two cells and no current can flow. Use a smaller size=."
        )

    # nodes at cell centres, indexed (i, j, k) -> i*ny*nz + j*nz + k
    ijk = np.stack(np.meshgrid(*[np.arange(v) for v in n], indexing="ij"), axis=-1).reshape(-1, 3)
    nodes = lo + (ijk + 0.5) * d
    nid = np.arange(len(nodes)).reshape(tuple(n))

    cen, tan, ln, area, ends, axis = [], [], [], [], [], []
    for ax in range(3):
        if n[ax] < 2:
            continue
        a = tuple(slice(0, v - 1) if i == ax else slice(None) for i, v in enumerate(n))
        b = tuple(slice(1, v) if i == ax else slice(None) for i, v in enumerate(n))
        na, nb = nid[a].reshape(-1), nid[b].reshape(-1)
        pa, pb = nodes[na], nodes[nb]
        k = len(na)
        u = np.zeros((k, 3))
        u[:, ax] = 1.0
        w, t = [d[i] for i in range(3) if i != ax]
        cen.append(0.5 * (pa + pb))
        tan.append(u)
        ln.append(np.full(k, d[ax]))
        area.append(np.full(k, w * t))
        ends.append((pa, pb))
        axis.append(np.full(k, ax))
    if not cen:
        raise ValueError("peec.bar_filaments: the lattice has no bar in any direction.")

    cen = np.concatenate(cen)
    tan = np.concatenate(tan)
    ln = np.concatenate(ln)
    area = np.concatenate(area)
    axis = np.concatenate(axis)
    nb = len(ln)

    gx, gw = np.polynomial.legendre.leggauss(int(quad))
    pos = (cen[:, None, :] + 0.5 * ln[:, None, None] * gx[None, :, None] * tan[:, None, :]).reshape(-1, 3)
    mom = (tan[:, None, :] * (ln[:, None] * gw[None, :] * 0.5)[:, :, None]).reshape(-1, 3)
    group = np.repeat(np.arange(nb), int(quad))
    wt = np.stack([np.array([d[i] for i in range(3) if i != ax]) for ax in axis])
    self_g = np.asarray(bar_self(jnp.asarray(ln), jnp.asarray(wt[:, 0]), jnp.asarray(wt[:, 1])))

    inc = np.zeros((len(nodes), nb))
    off = 0
    for (pa, pb), ax in zip(ends, sorted(set(axis.tolist()))):
        k = len(pa)
        a = tuple(slice(0, v - 1) if i == ax else slice(None) for i, v in enumerate(n))
        b = tuple(slice(1, v) if i == ax else slice(None) for i, v in enumerate(n))
        inc[nid[a].reshape(-1), np.arange(off, off + k)] = 1.0
        inc[nid[b].reshape(-1), np.arange(off, off + k)] = -1.0
        off += k

    return Filaments(
        jnp.asarray(pos),
        jnp.asarray(mom),
        jnp.asarray(self_g),
        group,
        jnp.asarray(inc),
        jnp.asarray(ln),
        jnp.asarray(area),
        jnp.asarray(nodes),
        np.zeros(nb, dtype=int),
        {"n": tuple(int(v) for v in n), "d": tuple(float(v) for v in d), "axis": axis},
    )
