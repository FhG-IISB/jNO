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

import jax
import jax.experimental.sparse as jsparse
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
    "lattice_apply",
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


def solve_network(
    fil: Filaments,
    sigma,
    terminals,
    sources,
    grounds=(),
    currents=(),
    omega=0.0,
    mu0=4e-7 * np.pi,
    matrix_free=None,
    tol=1e-8,
):
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

    Args (continued):
        tol: relative residual the iterative path is driven to. Measured on 6,806 bars, the impedance
            is identical to nine digits from 1e-4 to 1e-11, while the time is not — 2.83 s, 1.77 s,
            2.88 s, 17.79 s — so the default sits where the curve is flat rather than at the tightest
            value it can reach. Tighten it if a problem needs it; the residual is checked either way.
        matrix_free: ``None`` decides by structure — a bar lattice is applied by FFT and solved by
            GMRES, anything else forms the dense operator and factors it. ``True``/``False`` force
            one. The dense path costs O(N^2) memory and O(N^3) time, so it is the small-network path;
            the lattice path costs O(N) and O(N log N) per apply.

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
    lat = getattr(fil, "lattice", None)
    welded = isinstance(lat, dict) and "welded" in lat
    has_lattice = lat is not None and (not welded or any(b[2] is not None for b in lat["welded"]))
    free_form = has_lattice if matrix_free is None else bool(matrix_free)
    if free_form and lat is None:
        raise ValueError(
            "peec.solve_network: matrix_free=True needs a lattice somewhere in the network, and these "
            "filaments have none. A polyline's filaments are not Toeplitz, so there is no structure to "
            "exploit and the dense path is the honest one."
        )
    if free_form:
        scale = mu0 / (4.0 * jnp.pi)
        lp_apply = (
            welded_apply(fil, lambda r: 1.0 / r, mu_scale=scale)
            if welded
            else lattice_apply(fil, lambda r: 1.0 / r, mu_scale=scale)
        )
        Z = None
    else:
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

    free = np.array([n for n in range(nn) if n not in owner], dtype=int)
    # The constraint block is assembled as TRIPLETS, not rows. It is current balance plus a handful of
    # port conditions, so it carries a few entries per row against ne + nn columns: on a 12k-bar
    # lattice the incidence alone is 24,548 nonzeros in 61,271,808 slots -- 0.04 % -- and holding that
    # densely costs 1.4 GB and an 86-million-op matvec per Krylov step, next to a 3.7 ms FFT apply.
    An = np.asarray(A)
    rr, cc, vv, rhs = [], [], [], [np.zeros(ne, dtype=complex)]
    r0 = 0
    for n in free:  # current balance away from the terminals
        k = np.flatnonzero(An[n])
        rr.append(np.full(k.size, r0))
        cc.append(k)
        vv.append(An[n, k])
        r0 += 1
        rhs.append(np.zeros(1, dtype=complex))
    for t in names:  # a terminal is equipotential: tie its nodes to its first
        ids = idx[t]
        for n in ids[1:].tolist():
            rr.append(np.array([r0, r0]))
            cc.append(np.array([ne + n, ne + int(ids[0])]))
            vv.append(np.array([1.0, -1.0]))
            r0 += 1
            rhs.append(np.zeros(1, dtype=complex))
    for t in names:  # and one row for the terminal itself
        kind, *rest = row[t][1] if t in row else ("open", t, 0.0 + 0j)
        if kind == "source":
            a, b_, g = rest
            rr.append(np.array([r0, r0]))
            cc.append(np.array([ne + int(idx[a][0]), ne + int(idx[b_][0])]))
            vv.append(np.array([1.0, -1.0]))
        elif kind == "ground":
            _t, g = rest
            rr.append(np.array([r0]))
            cc.append(np.array([ne + int(idx[t][0])]))
            vv.append(np.array([1.0]))
        else:  # a fixed injected current, or an open terminal (which is zero injected current)
            _t, g = rest
            col = An[idx[t]].sum(0)
            k = np.flatnonzero(col)
            rr.append(np.full(k.size, r0))
            cc.append(k)
            vv.append(col[k])
        rhs.append(np.atleast_1d(np.asarray(g, dtype=complex)))
        r0 += 1
    if r0 != nn:
        raise ValueError(f"jno.peec: built {r0} constraint rows, expected {nn}; this is a bug.")
    crow, ccol, cval = np.concatenate(rr), np.concatenate(cc), np.concatenate(vv).astype(complex)
    b = jnp.asarray(np.concatenate(rhs))
    _refuse_disconnected(np.asarray(A), idx, sources, grounds, currents)

    if free_form:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        w = float(omega)
        Rc = R.astype(complex)
        zdiag = Rc + (1j * w) * jnp.asarray(_lattice_diag(fil, mu0))

        C = sp.coo_matrix((cval, (crow, ccol)), shape=(nn, ne + nn)).tocsr()
        CI, Cp = C[:, :ne], C[:, ne:]
        Cj = _bcoo(cval, crow, ccol, (nn, ne + nn))
        el = ccol < ne  # the current columns of the constraint block, for the preconditioner
        CIj = _bcoo(cval[el], crow[el], ccol[el], (nn, ne))
        Ac = A.T.astype(complex)

        # The Schur complement of the diagonal of Z. Sparse throughout: CI is current balance, A is
        # the incidence, and their product is graph-Laplacian shaped, so forming and factoring it
        # costs a fraction of the dense route (measured on 12k bars: 6.5 s + 21.1 s dense, against a
        # single FFT apply of 3.7 ms). The factorisation is host-side and eager -- a preconditioner
        # only has to accelerate, so nothing differentiable passes through it; the gradient comes from
        # the outer custom_linear_solve.
        Asp = sp.coo_matrix((An[An != 0], np.nonzero(An)), shape=(nn, ne)).tocsr()
        Dinv = sp.diags(1.0 / np.asarray(zdiag))
        S = (CI @ Dinv @ Asp.T.tocsr() + Cp).tocsc()
        lu = spla.splu(S)
        shape_out = jax.ShapeDtypeStruct((nn,), jnp.complex128)

        def _host_solve(r):
            return lu.solve(np.asarray(r)).astype(np.complex128)

        def M_apply(x):
            cur, phi = x[:ne], x[ne:]
            return jnp.concatenate([Rc * cur + (1j * w) * lp_apply(cur) - Ac @ phi, Cj @ x])

        def P_inv(r):
            ri, rp = r[:ne], r[ne:]
            rhs_p = rp - CIj @ (ri / zdiag)
            dphi = jax.pure_callback(_host_solve, shape_out, rhs_p)
            return jnp.concatenate([(ri + Ac @ dphi) / zdiag, dphi])

        # A SHORT restart. With the block preconditioner this system converges in a handful of steps,
        # and jax builds the whole Krylov basis whether or not it is needed: measured at 6,806 bars,
        # restart 200 took 10.07 s, restart 60 took 2.88 s, restart 20 took 2.14 s — all returning the
        # same impedance to seven digits. The basis was the cost, not the convergence.
        x, _ = jax.scipy.sparse.linalg.gmres(
            M_apply, b, M=P_inv, tol=float(tol), atol=0.0, restart=min(30, ne + nn), maxiter=400
        )
        resid = jnp.linalg.norm(M_apply(x) - b) / jnp.maximum(jnp.linalg.norm(b), 1e-300)
        if not bool(resid < max(1e2 * float(tol), 1e-9)):
            raise ValueError(
                f"jno.peec: the matrix-free solve did not converge (relative residual {float(resid):.2e}). "
                "Report it rather than trusting the numbers; the dense path is available as "
                "matrix_free=False for a network small enough to form."
            )
    else:
        top = jnp.concatenate([Z, -A.T.astype(complex)], axis=1)
        bot = jnp.zeros((nn, ne + nn), dtype=complex).at[crow, ccol].add(jnp.asarray(cval))
        M = jnp.concatenate([top, bot], axis=0)
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


def bar_filaments(shape, size=None, quad: int = 3, sigma=None):
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
        shape: a solid ``Shape``, or a SEQUENCE of them. A sequence shares ONE grid, which is the
            whole point: separate lattices couple through a block that is not Toeplitz, so several
            conductors on one grid stay a single FFT while several grids do not. Coplanar traces of
            equal thickness are exactly the case this is for.
        size: cell pitch — one number, or one per axis. Defaults to the shape's own ``size=``.

            A per-axis pitch is what makes a real conductor affordable. A power-module trace is
            0.57 mm thick on a 96.9 mm plate, so an isotropic grid fine enough to resolve the skin
            depth through the thickness spends the same resolution across the width, where nothing
            varies: 33.6 million bars against 137 thousand for ``(1.0, 1.0, 0.065)`` at the same
            through-thickness resolution.
        quad: Gauss points along each bar.
        sigma: conductivity per conductor. Needed only when conductors touch: a bar straddling two of
            them has half its length in each, so its conductivity is their series (harmonic) mean.

    Returns:
        :class:`Filaments`, with ``lattice`` describing the grid the FFT path needs.

    The cell count per axis is ``max(1, round(extent / size))``, so the pitch is the extent divided
    by a whole number rather than ``size`` exactly -- a lattice has to close on the box.
    """
    shapes = list(shape) if isinstance(shape, (list, tuple)) else [shape]
    declared = [sh._size for sh in shapes if sh._size is not None]
    # several conductors share one grid, so they share one pitch: the finest asked for, per axis.
    h = (
        size
        if size is not None
        else (
            np.min([np.broadcast_to(np.asarray(v, float).reshape(-1), (3,)) for v in declared], axis=0)
            if declared
            else None
        )
    )
    h = np.broadcast_to(np.asarray(h, dtype=float).reshape(-1), (3,)) if h is not None else np.zeros(3)
    if np.any(h <= 0):
        raise ValueError(
            "peec.bar_filaments: no cell pitch. Pass size=, or give the Shape a size= when you build "
            "it — a lattice count cannot be guessed from the geometry alone."
        )
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for sh in shapes:
        bnd = np.asarray(sh.bounds(), dtype=float).reshape(2, -1)
        b0, b1 = np.zeros(3), np.zeros(3)
        b0[: bnd.shape[1]], b1[: bnd.shape[1]] = bnd[0], bnd[1]
        lo, hi = np.minimum(lo, b0), np.maximum(hi, b1)
    ext = hi - lo
    ext = np.where(ext > 0, ext, h)  # a flat axis is one cell thick
    n = np.maximum(1, np.round(ext / h).astype(int))
    d = ext / n  # cell pitch per axis, closing exactly on the box
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError(f"peec.bar_filaments: the box has a zero or non-finite extent {tuple(ext)}.")
    if int(n.max()) < 2:
        raise ValueError(
            f"peec.bar_filaments: a pitch of {h} puts one cell across every axis of this box, so no bar "
            "joins two cells and no current can flow. Use a smaller size=."
        )

    # A regular grid over the BOUNDING BOX, then a mask: the grid deliberately ignores the shape, so
    # an L-shaped trace or a slot is a pattern of absent cells rather than a distorted mesh. That is
    # what keeps the operator translation-invariant, and therefore what keeps the FFT applicable.
    ijk = np.stack(np.meshgrid(*[np.arange(v) for v in n], indexing="ij"), axis=-1).reshape(-1, 3)
    centres = lo + (ijk + 0.5) * d
    # which conductor owns each cell: the first one containing it, matching how Shape.regions resolves
    own = np.full(len(centres), -1, dtype=int)
    for si, sh in enumerate(shapes):
        inside = np.asarray(sh.contains(centres)).reshape(-1)
        own = np.where((own < 0) & inside, si, own)
    keep = own >= 0
    if not keep.any():
        raise ValueError(
            f"peec.bar_filaments: a pitch of {tuple(h)} put no cell centre inside this conductor. Use a "
            "smaller "
            "size=; a cell has to fit within the geometry for the lattice to see it at all."
        )
    nid = np.full(int(np.prod(n)), -1)
    nid[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
    nid = nid.reshape(tuple(n))
    nodes = centres[keep]
    cell_part = own[keep]

    cen, tan, ln, area, ends, axis, owner, masks = [], [], [], [], [], [], [], {}
    for ax in range(3):
        if n[ax] < 2:
            continue
        a = tuple(slice(0, v - 1) if i == ax else slice(None) for i, v in enumerate(n))
        b = tuple(slice(1, v) if i == ax else slice(None) for i, v in enumerate(n))
        ia, ib = nid[a], nid[b]
        live = (ia >= 0) & (ib >= 0)  # a bar exists only where BOTH cells are metal
        if not live.any():
            continue
        masks[ax] = live
        na, nb = ia[live].reshape(-1), ib[live].reshape(-1)
        pa, pb = nodes[na], nodes[nb]
        k = len(na)
        u = np.zeros((k, 3))
        u[:, ax] = 1.0
        w, t = [d[i] for i in range(3) if i != ax]
        cen.append(0.5 * (pa + pb))
        tan.append(u)
        ln.append(np.full(k, d[ax]))
        area.append(np.full(k, w * t))
        ends.append((na, nb))
        axis.append(np.full(k, ax))
        owner.append(np.stack([cell_part[na], cell_part[nb]], axis=1))
    if not cen:
        raise ValueError(
            "peec.bar_filaments: no bar joins two cells of this conductor, so no current can flow. Use a smaller size=."
        )

    cen = np.concatenate(cen)
    tan = np.concatenate(tan)
    ln = np.concatenate(ln)
    area = np.concatenate(area)
    axis = np.concatenate(axis)
    owner = np.concatenate(owner)
    part = owner[:, 0]
    nb = len(ln)
    # A bar may straddle two conductors -- a shorting strap touching two plates is the normal case,
    # not an error. Its two halves are in SERIES, so the bar's conductivity is the harmonic mean; for
    # one material that degenerates to the material, and only a genuine mismatch changes anything.
    bar_sigma = None
    if sigma is not None:
        sg = np.asarray(sigma, dtype=float)
        if sg.size != len(shapes):
            raise ValueError(f"peec.bar_filaments: {sg.size} conductivities for {len(shapes)} conductors.")
        s0, s1 = sg[owner[:, 0]], sg[owner[:, 1]]
        bar_sigma = 2.0 * s0 * s1 / (s0 + s1)
    elif not np.all(owner[:, 0] == owner[:, 1]):
        bad = int(np.flatnonzero(owner[:, 0] != owner[:, 1])[0])
        raise ValueError(
            f"peec.bar_filaments: a bar joins cells of conductors {owner[bad, 0]} and {owner[bad, 1]}, so "
            "its conductivity depends on both. Pass sigma= (one per conductor) and it is resolved as the "
            "series combination."
        )

    gx, gw = np.polynomial.legendre.leggauss(int(quad))
    pos = (cen[:, None, :] + 0.5 * ln[:, None, None] * gx[None, :, None] * tan[:, None, :]).reshape(-1, 3)
    mom = (tan[:, None, :] * (ln[:, None] * gw[None, :] * 0.5)[:, :, None]).reshape(-1, 3)
    group = np.repeat(np.arange(nb), int(quad))
    wt = np.stack([np.array([d[i] for i in range(3) if i != ax]) for ax in axis])
    self_g = np.asarray(bar_self(jnp.asarray(ln), jnp.asarray(wt[:, 0]), jnp.asarray(wt[:, 1])))

    inc = np.zeros((len(nodes), nb))
    off = 0
    for na, nb_ in ends:
        k = len(na)
        inc[na, np.arange(off, off + k)] = 1.0
        inc[nb_, np.arange(off, off + k)] = -1.0
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
        part,
        {
            "n": tuple(int(v) for v in n),
            "d": tuple(float(v) for v in d),
            "axis": axis,
            "masks": masks,
            "sigma": bar_sigma,
        },
    )


def lattice_apply(fil: Filaments, g, mu_scale: float = 1.0, quad: int = 3):
    """``apply(cur) -> Lp @ cur`` for a bar lattice, by FFT, without forming ``Lp``.

    Two facts about a bar lattice make this exact rather than approximate:

    * a bar's current runs along one axis, and ``mom_i . mom_j`` vanishes between perpendicular
      bars, so ``Lp`` is block diagonal by direction and each family can be applied on its own;
    * within a family the bars sit on a regular grid and the kernel depends only on their
      separation, so the block is block-Toeplitz — :func:`~jno.utils.solver.kernel.lattice_operator`
      applies it in ``O(N log N)`` and ``O(N)`` memory instead of ``O(N^2)`` of both.

    The sub-point quadrature goes into the generator (every bar of a family carries the same offsets,
    so the double sum still depends only on the separation), which is what makes this agree with
    :func:`~jno.utils.solver.kernel.pair_matrix` to round-off rather than to a one-point rule.

    Args:
        fil: filaments carrying a ``lattice`` description, from :func:`bar_filaments`.
        g: the kernel, a function of distance.
        mu_scale: a constant multiplying the result (``mu0 / 4 pi`` for partial inductance).
        quad: sub-points per bar; must match the discretisation's own.
    """
    from .kernel import lattice_operator

    lat = getattr(fil, "lattice", None)
    if lat is None:
        raise ValueError(
            "peec.lattice_apply: these filaments do not sit on a lattice. Only a bar lattice from "
            "bar_filaments is block-Toeplitz; a polyline's filaments are not, and welding several "
            "conductors together breaks the structure even when each part had it."
        )
    n, d, axis = lat["n"], lat["d"], np.asarray(lat["axis"])
    ln = np.asarray(fil.length)
    gx, gw = np.polynomial.legendre.leggauss(int(quad))

    masks = lat.get("masks") or {}
    ops, slices, shapes, where = [], [], [], []
    start = 0
    for ax in sorted(set(axis.tolist())):
        k = int((axis == ax).sum())
        shape = tuple(v - 1 if i == ax else v for i, v in enumerate(n))
        length = float(ln[axis == ax][0])
        sub = np.zeros((int(quad), 3))
        sub[:, ax] = 0.5 * length * gx
        w = length * gw * 0.5
        sg = float(np.asarray(fil.self_g)[axis == ax][0])
        ops.append(lattice_operator(shape, d, g, sg, sub=sub, w=w))
        slices.append(slice(start, start + k))
        shapes.append(shape)
        m = masks.get(ax)
        where.append(None if m is None else jnp.asarray(np.flatnonzero(np.asarray(m).reshape(-1))))
        start += k
    if start != len(ln):
        raise ValueError(f"peec.lattice_apply: the families cover {start} of {len(ln)} bars; this is a bug.")

    def real_apply(cur):
        out = []
        for op, sl, sh, idx in zip(ops, slices, shapes, where):
            x = cur[sl]
            if idx is None:
                out.append(op(x.reshape(sh)).reshape(-1))
            else:
                # Absent cells carry no current, so scatter into the FULL family grid, apply, and read
                # back only the live slots. Masking this way is what lets a hole or an L-shape keep the
                # translation invariance the FFT needs — the grid stays full, the current does not.
                # unique/sorted are true by construction (flatnonzero), and saying so is what makes the
                # scatter transposable — reverse mode refuses an unproven one.
                full = jnp.zeros(int(np.prod(sh)), x.dtype).at[idx].set(x, unique_indices=True, indices_are_sorted=True)
                out.append(op(full.reshape(sh)).reshape(-1)[idx])
        return jnp.concatenate(out)

    def apply(cur):
        cur = jnp.asarray(cur)
        # Lp is real, and the circulant embedding uses rfftn, which will not take a complex array.
        # A complex current is therefore applied through its parts: Lp(a + ib) = Lp a + i Lp b.
        if jnp.iscomplexobj(cur):
            return mu_scale * (real_apply(jnp.real(cur)) + 1j * real_apply(jnp.imag(cur)))
        return mu_scale * real_apply(cur)

    return apply


def _lattice_diag(fil: Filaments, mu0: float):
    """``Lp_aa`` for every bar — the diagonal a Jacobi preconditioner needs, without forming ``Lp``."""
    mom = np.asarray(fil.mom)
    grp = np.asarray(fil.group)
    tot = np.zeros((int(grp.max()) + 1, mom.shape[1]))
    np.add.at(tot, grp, mom)
    return np.asarray(fil.self_g) * (tot * tot).sum(1) * (mu0 / (4.0 * np.pi))


def _sub_slice(fil, lo, hi):
    """The sub-point rows of ``fil`` belonging to elements ``[lo, hi)``, and their local group labels."""
    grp = np.asarray(fil.group)
    m = (grp >= lo) & (grp < hi)
    return jnp.asarray(np.flatnonzero(m)), jnp.asarray(grp[m] - lo), int(hi - lo)


def cross_apply(pos_a, mom_a, grp_a, na, pos_b, mom_b, grp_b, nb, g, scale=1.0, chunk=4096):
    """``x -> Lp_ab @ x`` and its transpose, without forming the rectangular block.

    Two conductors on different discretisations couple through a block that is neither Toeplitz nor
    square, so neither the FFT nor a lattice trick applies to it. It does not need them: one side is
    small (a bond wire is a few hundred filaments against a trace's hundred thousand), so the block is
    thin and evaluating it per apply costs less than storing it.
    """
    pa, ma = jnp.asarray(pos_a), jnp.asarray(mom_a)
    pb, mb = jnp.asarray(pos_b), jnp.asarray(mom_b)
    ga, gb = jnp.asarray(grp_a), jnp.asarray(grp_b)

    def forward(x):  # (nb,) -> (na,)
        xb = jnp.asarray(x)[gb]  # spread the element current onto its sub-points
        out = jnp.zeros(pa.shape[0], xb.dtype)
        for lo in range(0, pa.shape[0], chunk):
            hi = min(lo + chunk, pa.shape[0])
            d = pa[lo:hi, None, :] - pb[None, :, :]
            r = jnp.sqrt(jnp.clip((d * d).sum(-1), 1e-300))
            out = out.at[lo:hi].set(((ma[lo:hi] @ mb.T) * g(r)) @ xb)
        return scale * jax.ops.segment_sum(out, ga, num_segments=na)

    def transpose(y):  # (na,) -> (nb,)
        ya = jnp.asarray(y)[ga]
        out = jnp.zeros(pb.shape[0], ya.dtype)
        for lo in range(0, pa.shape[0], chunk):
            hi = min(lo + chunk, pa.shape[0])
            d = pa[lo:hi, None, :] - pb[None, :, :]
            r = jnp.sqrt(jnp.clip((d * d).sum(-1), 1e-300))
            out = out + ((ma[lo:hi] @ mb.T) * g(r)).T @ ya[lo:hi]
        return scale * jax.ops.segment_sum(out, gb, num_segments=nb)

    return forward, transpose


def welded_apply(fil: Filaments, g, mu_scale: float = 1.0, quad: int = 3):
    """``apply(cur) -> Lp @ cur`` for a network of several discretisations, still without forming Lp.

    Each block keeps whatever structure it has -- a lattice by FFT, a set of filaments densely -- and
    the blocks couple through :func:`cross_apply`. So a trace layer of a hundred thousand bars stays
    O(N) while the bond wires landing on it stay exact.
    """
    from .kernel import pair_matrix

    blocks = getattr(fil, "lattice", None)
    if not (isinstance(blocks, dict) and "welded" in blocks):
        raise ValueError("peec.welded_apply: these filaments are not a welded network.")
    spans = blocks["welded"]

    diag, sel = [], []
    for lo, hi, lat in spans:
        rows, gl, cnt = _sub_slice(fil, lo, hi)
        sub = Filaments(
            jnp.asarray(fil.pos)[rows],
            jnp.asarray(fil.mom)[rows],
            jnp.asarray(fil.self_g)[lo:hi],
            np.asarray(gl),
            None,
            jnp.asarray(fil.length)[lo:hi],
            jnp.asarray(fil.area)[lo:hi],
            None,
            None,
            lat,
        )
        if lat is None:
            k = pair_matrix(sub.pos, sub.mom, g, sub.self_g, group=sub.group) * mu_scale
            diag.append(lambda x, k=k: k @ x)
        else:
            diag.append(lattice_apply(sub, g, mu_scale=mu_scale, quad=quad))
        sel.append((lo, hi, rows, gl, cnt))

    cross = {}
    for i, (loi, hii, ri, gi, ci) in enumerate(sel):
        for j, (loj, hij, rj, gj, cj) in enumerate(sel):
            if j <= i:
                continue
            cross[(i, j)] = cross_apply(
                jnp.asarray(fil.pos)[ri],
                jnp.asarray(fil.mom)[ri],
                gi,
                ci,
                jnp.asarray(fil.pos)[rj],
                jnp.asarray(fil.mom)[rj],
                gj,
                cj,
                g,
                scale=mu_scale,
            )

    def apply(cur):
        cur = jnp.asarray(cur)
        out = [d(cur[lo:hi]) for d, (lo, hi, *_r) in zip(diag, sel)]
        for (i, j), (fwd, tr) in cross.items():
            out[i] = out[i] + fwd(cur[sel[j][0] : sel[j][1]])
            out[j] = out[j] + tr(cur[sel[i][0] : sel[i][1]])
        return jnp.concatenate(out)

    return apply


def _bcoo(val, row, col, shape):
    """A BCOO from COO triplets, for a matvec inside a traced body."""
    return jsparse.BCOO((jnp.asarray(val), jnp.stack([jnp.asarray(row), jnp.asarray(col)], axis=1)), shape=shape)
