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

import functools
import inspect
import logging
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .kernel import bar_self, internal_impedance, slab_transfer_impedance, wire_self

__all__ = [
    "Filaments",
    "line_filaments",
    "bar_filaments",
    "network_impedance",
    "port_spec",
    "terminal_nodes",
    "solve_network",
    "lattice_apply",
    "near_block",
    "element_centres",
    "resolve_sigma",
]


class Filaments(NamedTuple):
    """A conductor discretised into straight filaments, shaped for the kernel and for a circuit solve.

    ``pos``/``mom``/``self_g``/``group`` go straight to :func:`~jno.utils.solver.kernel.pair_quadratic`.
    ``incidence`` and ``length`` are what a network solve needs on top: which nodes each filament
    joins, and how long it is.

    The incidence is a ``scipy.sparse`` matrix, and it has to be: a filament touches two nodes, so
    the array is two nonzeros per column, and holding it densely is what the size of a real problem
    goes into. On the example power module -- 137,350 bars over 55,000 nodes -- dense is 60 GB and
    sparse is a few MB.
    """

    pos: object  # (N*quad, 3) sub-point positions
    mom: object  # (N*quad, 3) tangent * length * Gauss weight
    self_g: object  # (N,)      per-filament self term
    group: object  # (N*quad,) filament label for each sub-point
    incidence: object  # (n_node, N) SPARSE: +1 where a filament leaves a node, -1 where it enters
    length: object  # (N,)
    area: object  # (N,)    conducting cross-section, which is all the solve needs of the shape
    nodes: object  # (n_node, 3) node positions, which is how a port is addressed
    part: object  # (N,)    index of the shape each filament came from, so per-conductor data can be spread
    skin: object  # (N,)    transverse size the skin effect acts over: a wire's radius, a bar's thickness
    round_: object  # (N,) bool: a round section takes the cylindrical internal impedance, a bar the slab one
    span: object  # (N,) int: elements across the conductor's THICKNESS here; 1 = this element is all of it
    lattice: object = None  # grid description when the elements sit on one, which is what the FFT path needs
    # (N,) int: the element carrying the OTHER face of the same slab, -1 where there is none. A
    # conductor thick against the skin depth carries a current sheet per face rather than one
    # current spread through it -- see `slab_transfer_impedance`.
    pair: object = None


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


def element_centres(fil: "Filaments") -> jnp.ndarray:
    """Each element's midpoint, ``(n_elements, 3)``.

    Read back from the quadrature points rather than stored alongside them: Gauss-Legendre abscissae
    are symmetric about their interval, so their unweighted mean IS the midpoint -- exactly, not
    approximately, for the straight elements both discretisations produce. Doing it this way keeps
    the result differentiable and correct when the positions are themselves traced.
    """
    ne = fil.incidence.shape[1]
    grp = jnp.asarray(fil.group)
    cnt = jax.ops.segment_sum(jnp.ones(grp.shape[0]), grp, ne)
    return jax.ops.segment_sum(fil.pos, grp, ne) / cnt[:, None]


def _is_field(value) -> bool:
    """Whether an attached value is a FUNCTION of position rather than a value.

    ``isroutine`` for the same reason :meth:`jno.Domain._resolve_attached` uses it -- a symbolic
    expression defines ``__call__`` (that is how ``u(x, y)`` binds), so a bare ``callable`` test
    would try to invoke one as a field. ``functools.partial`` is admitted on top because it is the
    one ordinary way a field arrives already carrying its parameters.
    """
    return inspect.isroutine(value) or isinstance(value, functools.partial)


def _field_arity(fn) -> int:
    """How many coordinates ``fn`` wants: ``f(x)``, ``f(x, y)`` or ``f(x, y, z)``.

    A planar density is the common case -- a trace is thin, and its material varies across the board
    but not through the 0.57 mm thickness -- so ``lambda x, y: ...`` has to mean what it says rather
    than being an arity error. Anything variadic or un-introspectable gets all three.
    """
    try:
        ps = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return 3
    if any(p.kind is p.VAR_POSITIONAL for p in ps):
        return 3
    n = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty for p in ps)
    return 3 if n == 0 or n > 3 else n


def _is_anisotropic(value, n: int, what: str) -> bool:
    """Is this the ``(sx, sy, sz)`` spelling rather than one of the scalar ones?

    A tuple of three, or an array shaped ``(3,)`` or ``(n, 3)``. The one case that cannot be read
    off the shape is ``(3,)`` on a conductor with exactly three elements, which is either three
    components or three elements -- so it is refused rather than guessed.
    """
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            return True
        raise ValueError(
            f"peec: the conductivity attached to {what} is a sequence of {len(value)}. A tuple means "
            "the ANISOTROPIC spelling `sigma=(sx, sy, sz)` and must have exactly three entries; for "
            "one value per element pass an array instead."
        )
    if _is_field(value):
        return False
    shape = np.shape(value)
    if shape == (n, 3):
        return True
    if shape == (3,):
        if n == 3:
            raise ValueError(
                f"peec: the conductivity attached to {what} has shape (3,) and the conductor has "
                "exactly 3 elements, so this is ambiguous -- three components, or three elements? "
                "Spell it: `sigma=(sx, sy, sz)` for the anisotropic one, or reshape to (3, 1) and "
                "broadcast for one value per element."
            )
        return True
    return False


def resolve_sigma(value, xyz, what: str, tangent=None):
    """Resolve one conductor's attached conductivity onto ``len(xyz)`` positions.

    A conductivity is a design variable as often as it is a material constant, and it arrives in
    three spellings, all of which land here as one vector:

    * a **scalar** (a tracer is fine) -- one conductivity for the whole conductor, which is a
      material, ``sigma(T)`` in an electro-thermal loop, or a single scaling knob;
    * a **callable of position** -- ``f(x, y, z)``, ``f(x, y)`` or ``f(x)``, evaluated at each
      element. This is the resolution-independent spelling: it says nothing about the pitch, so the
      same design survives a change of ``size=``, which a per-element vector cannot;
    * a **vector**, already one value per element, for a design variable that IS the discretisation.

    Positions are host coordinates and the returned values may be traced, which is the split that
    matters: the geometry is structural, the material on it is what a gradient flows back to.

    Args:
        value: the attached conductivity, in any of the three forms above.
        xyz: ``(n, 3)`` positions -- cell centres for a lattice, element midpoints for a line.
        what: the conductor's name, for the error messages.

    A fourth spelling makes the material **anisotropic**: ``(sx, sy, sz)``, where each component may
    itself be any of the three above. The lattice already carries one current direction per bar
    family, so a component only has to reach the family it belongs to.

    Scope, up front: **diagonal** anisotropy only. The bars are axis-aligned, so an off-diagonal
    conductivity has nowhere to live in this discretisation. A wire is one-dimensional, so what
    reaches it is the component along its own tangent, ``t . sigma . t`` -- pass ``tangent``.

    Args:
        value: the attached conductivity, in any of the four forms above.
        xyz: ``(n, 3)`` positions -- cell centres for a lattice, element midpoints for a line.
        what: the conductor's name, for the error messages.
        tangent: ``(n, 3)`` element directions. Given, an anisotropic value is contracted onto them
            and a ``(n,)`` vector comes back; absent, the ``(n, 3)`` components are returned and the
            caller picks per axis family (which is what a lattice does).

    Returns:
        ``(n,)``, or ``(n, 3)`` for an anisotropic value with no ``tangent``.
    """
    n = int(np.shape(xyz)[0])
    if _is_anisotropic(value, n, what):
        cols = [jnp.asarray(resolve_sigma(v, xyz, f"{what} (component {ax})")) for ax, v in zip("xyz", value)]
        out = jnp.stack(cols, axis=1)
        if tangent is None:
            return out
        # A filament carries one current along its own direction, and for a diagonal sigma the
        # conductivity along a unit t is `t . sigma . t = sum_i t_i^2 sigma_i`. Transverse components
        # cannot drive a current in a one-dimensional conductor, and this is what says so.
        tan = jnp.asarray(tangent, dtype=out.dtype)
        tan = tan / jnp.maximum(jnp.linalg.norm(tan, axis=1, keepdims=True), 1e-300)
        return jnp.sum(out * tan**2, axis=1)
    if _is_field(value):
        k = _field_arity(value)
        out = jnp.asarray(value(*[jnp.asarray(xyz)[:, i] for i in range(k)]))
        if out.ndim == 0:
            return jnp.broadcast_to(out, (n,))
        if out.shape != (n,):
            raise ValueError(
                f"peec: the conductivity attached to {what} is a function of position, so it has to "
                f"return one value per element -- {n} of them, or a scalar. It returned shape "
                f"{tuple(out.shape)}. It is called with whole coordinate ARRAYS, not one point at a "
                "time, so write it in terms that broadcast (jno.np / jnp), not a Python scalar branch."
            )
        return out
    arr = jnp.asarray(value)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (n,))
    if arr.shape != (n,):
        raise ValueError(
            f"peec: {arr.size} conductivities were attached to {what}, which discretises into {n} "
            "elements. Give one value per element, a single scalar, or -- better for a design "
            "variable -- a callable of position, `sigma=lambda x, y, z: ...`, which does not depend "
            "on the pitch and so survives a change of size=."
        )
    return arr


def line_filaments(shape, size: float = None, quad: int = 3, points=None, radii=None, quad_t: int = 1):
    """Discretise :meth:`jno.Shape.line` conductors into filaments, with Gauss sub-points.

    Args:
        shape: a ``Shape`` whose plan is a single ``Line`` leaf, or a sequence of them. A sequence
            becomes ONE network: lines that share an endpoint share a node, which is how a branch,
            a tee or a parallel pair is expressed -- there is no separate "join" step.
        size: target filament length. Defaults to each shape's own ``size=``.
        quad: Gauss points per filament. One point is 7.8 % low against a closed form on the worst
            case (collinear neighbours); 2 gives 2.5 %, 3 gives 1.2 %, 8 gives 0.21 %. Three is the
            default because it is where the curve flattens against its cost.
        radii: optional replacement wire radius, one scalar per shape, which may be TRACED. A bond
            wire's gauge is a design variable in its own right -- a thicker wire lowers both the
            resistance and the self inductance -- and the realistic problem is distributing a fixed
            total cross-section (the assembly's cost) over the wires that need it. It enters the
            area, the self term and the skin depth, all of which were already jax.
        points: optional replacement polyline vertices, one ``(n_i, 3)`` array per shape, which may
            be TRACED. Everything geometric is then computed from them in jax, so a gradient flows
            back to the vertices -- the r-adaptivity contract, where the topology is fixed and only
            the positions move. What stays fixed is decided from the shape's OWN vertices: how many
            filaments a segment is cut into, and which endpoints are the same node. Move a vertex far
            enough to change either and the answer is for the topology you started with, silently --
            so use this to refine a routing, not to redesign one.

            ``nodes`` is then traced too, so :func:`terminal_nodes` -- which reads coordinates --
            must be called ONCE on the reference geometry and its indices reused. Which nodes a pad
            owns is structural, like the numbering it indexes into.

    Returns:
        :class:`Filaments`.

    Nodes are the filament endpoints, deduplicated onto a grid of ``1e-9 x`` the shortest filament.
    A CLOSED polyline therefore yields a loop rather than a chain -- which is what makes a current
    path possible at all -- and two lines meeting at a point are electrically joined. The grid is
    needed rather than exact equality because a shared vertex reached from its two segments differs
    in the last bits; it is nine orders below the element size, so it cannot fuse two terminals that
    are meant to stay apart.

    Each polyline is subdivided so that no filament exceeds ``size``, and each original vertex stays
    a filament boundary -- a bend must not fall inside a straight element.
    """
    shapes = list(shape) if isinstance(shape, (list, tuple)) else [shape]
    if points is not None and len(points) != len(shapes):
        raise ValueError(f"peec.line_filaments: {len(points)} point arrays for {len(shapes)} shapes.")
    if radii is not None and len(radii) != len(shapes):
        raise ValueError(f"peec.line_filaments: {len(radii)} radii for {len(shapes)} shapes.")

    # ---- host pass: the STRUCTURE, read off the shapes' own vertices --------------------------
    # Which segment each filament belongs to and where along it, so the geometry can be rebuilt from
    # any vertices later; and the node numbering, which a moving vertex must not change.
    verts, base, ia, ib, jj, kk, radii_h, part = [], 0, [], [], [], [], [], []
    ends = []
    for si, sh in enumerate(shapes):
        prim = _leaf(sh, "Line")
        h = float(size if size is not None else (sh._size if sh._size is not None else 0.0))
        if h <= 0:
            raise ValueError(
                "peec.line_filaments: no filament length. Pass size=, or give the Shape a size= when "
                "you build it -- a filament count cannot be guessed from the geometry alone."
            )
        P = np.asarray(prim.points, dtype=float).reshape(-1, 3)
        verts.append(P)
        for e, (a, d) in enumerate(zip(P[:-1], P[1:] - P[:-1])):
            ln0 = float(np.linalg.norm(d))
            if ln0 <= 0.0:
                continue
            k = max(1, int(np.ceil(ln0 / h)))  # vertices stay filament boundaries: subdivide within
            u, step = d / ln0, ln0 / k
            for j in range(k):
                ia.append(base + e)
                ib.append(base + e + 1)
                jj.append(j)
                kk.append(k)
                radii_h.append(prim.r)
                part.append(si)
                ends.append((a + u * (step * j), a + u * (step * (j + 1))))
        base += len(P)
    if not ends:
        raise ValueError("peec.line_filaments: the polyline has no segment longer than zero.")

    ia, ib = np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)
    jj, kk = np.asarray(jj, dtype=float), np.asarray(kk, dtype=float)
    # per FILAMENT, from the per-shape value; traced when the caller supplies one
    _which = np.asarray(part, dtype=int)
    rad = jnp.stack([jnp.asarray(v) for v in radii])[_which] if radii is not None else np.asarray(radii_h)
    n = len(ia)

    # nodes = filament endpoints snapped to a grid far below the element size
    tol = 1e-9 * float(min(np.linalg.norm(b - a) for a, b in ends))
    key, node_of, ir, ic, iv = {}, [], [], [], []
    for k, (a, b) in enumerate(ends):
        for pt, sign, off in ((a, +1.0, 0.0), (b, -1.0, 1.0)):
            t = tuple(np.round(np.asarray(pt) / tol).astype(np.int64).tolist())
            if t not in key:
                key[t] = len(node_of)
                node_of.append((k, off))  # one filament and which end of it defines this node
            ir.append(key[t])
            ic.append(k)
            iv.append(sign)
    nfil = np.asarray([f for f, _ in node_of], dtype=int)
    noff = np.asarray([o for _, o in node_of], dtype=float)

    # ---- jax pass: every GEOMETRIC quantity, from vertices that may be traced -----------------
    PTS = jnp.concatenate([jnp.asarray(v, dtype=float) for v in (points if points is not None else verts)])
    a3, b3 = PTS[ia], PTS[ib]
    d3 = b3 - a3
    seg = jnp.sqrt(jnp.sum(d3 * d3, axis=1))
    u3 = d3 / seg[:, None]
    ln = seg / kk
    cen = a3 + u3 * (ln * (jj + 0.5))[:, None]
    nodes = PTS[ia[nfil]] + u3[nfil] * (ln[nfil] * (jj[nfil] + noff))[:, None]

    # A filament is thin by construction, so it needs no TRANSVERSE sampling the way a lattice cell
    # does -- but a welded network mixes the two, and the vectorised near-field block requires one
    # sub-point count throughout. So a line gets the same COUNT, spent where it helps a wire: more
    # points along its own length.
    gx, gw = np.polynomial.legendre.leggauss(int(quad) * int(quad_t) ** 2)
    gx, gw = jnp.asarray(gx), jnp.asarray(gw)
    # sub-points along each filament, and moments that sum to `tangent * length` per filament
    pos = (cen[:, None, :] + 0.5 * ln[:, None, None] * gx[None, :, None] * u3[:, None, :]).reshape(-1, 3)
    mom = (u3[:, None, :] * (ln[:, None] * gw[None, :] * 0.5)[:, :, None]).reshape(-1, 3)
    group = np.repeat(np.arange(n), int(quad) * int(quad_t) ** 2)
    self_g = wire_self(ln, rad if isinstance(rad, jax.core.Tracer) else np.asarray(rad))
    area = jnp.pi * jnp.asarray(rad) ** 2

    return Filaments(
        pos,
        mom,
        self_g,
        group,
        sp.coo_matrix((iv, (ir, ic)), shape=(len(node_of), n)).tocsr(),
        ln,
        area,
        nodes,
        np.asarray(part, dtype=int),
        jnp.asarray(rad),
        np.ones(n, dtype=bool),
        np.ones(n, dtype=int),  # a wire element carries the whole cross-section, always
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

    # this path forms the dense operator anyway, so densifying the incidence changes nothing
    A = jnp.asarray(fil.incidence.toarray())
    nn, ne = A.shape
    sig = jnp.broadcast_to(jnp.asarray(sigma, dtype=float), (ne,))
    _check_unresolved_thickness(fil, sig, omega, mu0)
    R, _pidx, _cz = _element_impedance(fil, omega, sig, mu0)
    Lp = pair_matrix(fil.pos, fil.mom, lambda r: 1.0 / r, fil.self_g, group=fil.group) * (mu0 / (4.0 * jnp.pi))
    Z = jnp.diag(R.astype(complex)) + 1j * omega * Lp
    if _pidx is not None:  # the other face of the same slab
        Z = Z.at[jnp.arange(R.shape[0]), jnp.asarray(_pidx)].add(_cz)

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
    """Read a PEEC constraint list into ``(sources, currents, grounds, devices)``.

    Four forms, and only four::

        v(A) - v(B) - g        a source of g volts from terminal A to terminal B
        v(A) - g               terminal A held at g volts (g = 0 is the ground / reference)
        i(A) - g               terminal A carries g amps (g = 0 is an open terminal)
        v(A) - v(B) - Z*i(A)   a two-terminal DEVICE of impedance Z between A and B

    The last is Ohm's law, and it is written as one. It is how a component enters the network
    WITHOUT being a lump of metal: a MOSFET's on-resistance is a device parameter, so voxelising the
    die makes the answer follow the grid instead -- a 0.18 mm die on a 0.285 mm grid is 1.6x too
    resistive before the current-crowding error on top. A device carries no partial inductance and
    no coupling to anything, because it is a circuit element and not a piece of geometry.

    A relation across TWO regions only means something on the potential, so ``i(A) - i(B)`` is
    refused rather than guessed at.

    Pure: it reads names and constants off the trace and touches no geometry, so what a terminal
    NAME resolves to is the caller's business.
    """
    from jno._fem import _bare, _scalar_const
    from jno.trace.views import _tag_of_coord_vars

    sources, currents, grounds, devices = [], [], [], []
    for c in constraints:
        names = _trial_names(c)
        if names == {current, potential}:  # a device: the only form naming BOTH
            bare_d = _bare(c)
            tie_d, extra = getattr(c, "_periodic_tie", None), getattr(c, "_tie_extra", None)
            if tie_d is None or getattr(bare_d, "op", None) != "-":
                raise ValueError(
                    f"jno.peec: a constraint naming both {potential!r} and {current!r} is read as a "
                    f"two-terminal device, `{potential}(A) - {potential}(B) - Z*{current}(A)`. This one "
                    "is not in that form."
                )
            if tie_d[0] == tie_d[1]:
                raise ValueError(
                    f"jno.peec: a device from {tie_d[0]!r} to itself is a short across one terminal and "
                    f"has no reading. Name the two terminals it sits between: `{potential}(A) - "
                    f"{potential}(B) - Z*{current}(A)`."
                )
            if extra is not None and extra != tie_d[0]:
                # The current must be measured at the terminal the voltage is measured FROM, or the
                # sign is the reader's guess. The trace keeps only the pair plus this one newcomer, so
                # a current written anywhere else is refused rather than folded into a pair it is not
                # part of. (`extra is None` means the SAME bound terminal was reused for v and i,
                # which is the same thing said more tersely.)
                raise ValueError(
                    f"jno.peec: device {tie_d[0]!r}-{tie_d[1]!r} takes its current at {extra!r}. Write "
                    f"it at the terminal the voltage is measured from -- `{potential}({tie_d[0]}) - "
                    f"{potential}({tie_d[1]}) - Z*{current}({tie_d[0]})` -- so the sign of Z is the "
                    "passive one and not a guess."
                    + (
                        f" ({extra!r} is not one of this device's terminals at all, so what was written "
                        "is a controlled source; those are not supported.)"
                        if extra not in tie_d
                        else ""
                    )
                )
            devices.append((tie_d[0], tie_d[1], _device_impedance(bare_d.right, current, potential)))
            continue
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
    return sources, currents, grounds, devices


def _host_z(z, fallback=None):
    """A concrete stand-in for a device impedance, for the preconditioner only.

    The LIVE value first, always: it is what the operator carries, and the preconditioner should
    match it whenever it can be read. Only when it cannot -- a traced Z inside a jaxpr trace, which
    an electro-thermal fixed point produces -- does the DECLARED value stand in, and failing that an
    ideal device (``Z = 0``).

    Getting that precedence backwards is not harmless. Preferring the declared value even when the
    live one was readable left the preconditioner at the nominal design point while the operator
    moved with temperature, and the near-field preconditioner is doing heavy lifting here (11-30
    Krylov iterations against 3441 without it), so a few percent of drift stopped the solve
    converging at all: 1.9e-02 at a 10 K rise, 9.3e-02 at 50 K.
    """
    try:
        return complex(np.asarray(jax.lax.stop_gradient(jnp.asarray(z))))
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError, TypeError, ValueError):
        return 0.0 + 0j if fallback is None else complex(fallback)


def _device_impedance(node, current, potential):
    """The ``Z`` in ``Z * i(A)`` -- either factor order, and a bare ``i(A)`` is one ohm."""
    from jno._fem import _scalar_const

    if getattr(node, "op", None) == "*":
        for a, b in ((getattr(node, "left", None), getattr(node, "right", None)), (node.right, node.left)):
            z = _scalar_const(a) if a is not None else None
            if z is not None and _trial_names(b) == {current}:
                return complex(z)
    elif _trial_names(node) == {current}:
        return 1.0 + 0j
    raise ValueError(
        f"jno.peec: a device's impedance must be a CONSTANT times the terminal current, as in "
        f"`{potential}(A) - {potential}(B) - 5e-3*{current}(A)` for a 5 milliohm on-resistance, or "
        f"`(R + 1j*w*L)*{current}(A)` for a lumped R-L. A varying or field-dependent impedance is a "
        "conductor, so give it geometry instead."
    )


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


def _converged(resid, limit):
    if not (resid < limit):
        raise ValueError(
            f"jno.peec: the matrix-free solve did not converge (relative residual {resid:.2e}). "
            "Report it rather than trusting the numbers. A DEEPER RESTART is the usual fix, and it is "
            "often faster than the shallower one that failed, because it stops instead of exhausting "
            "its budget: `solve(restart=48)`, or higher. Failing that `solve(matrix_free=False)` forms "
            "the dense operator, exact but O(N^2) memory and so for a network small enough to form."
        )


def _finite(ok):
    if not ok:
        raise ValueError(
            "jno.peec: the circuit equations are singular — the solve returned non-finite currents. "
            "The usual cause is a part of the network that no port reaches, or a conductor with no "
            "path back to the reference."
        )


def _check(value, fn, *args):
    """Run a correctness guard on ``value``, eagerly OR under ``jit``.

    The house pattern elsewhere is to SKIP a concrete-only guard when the value is a tracer, which
    silently disarms it exactly when a long unattended optimisation is running. This solver is one
    that can diverge quietly -- a transpose solve reaching 1e+26 while the forward converged is a
    thing that has actually happened here -- so the guard is routed through ``jax.debug.callback``
    instead. That fires on the HOST with the runtime value, so it survives ``jit`` and still raises;
    the cost is one sync point per solve, which against a Krylov solve is nothing.
    """
    if isinstance(value, jax.core.Tracer):
        jax.debug.callback(lambda v, *a: fn(np.asarray(v).item() if np.ndim(v) == 0 else v, *a), value, *args)
    else:
        fn(np.asarray(value).item() if np.ndim(value) == 0 else value, *args)


#: Built-once-per-network geometry, bounded. Each entry holds its Filaments so the id cannot be
#: recycled underneath it, and the operator it keeps is the largest thing here -- two is plenty for a
#: sweep over one network, and a third would cost more memory than the rebuild it saves.
_GEOM_CACHE: dict = {}
_GEOM_CACHE_MAX = 2


def _frozen_geometry(fil) -> bool:
    """Whether the metal's position is a value, AND we are not inside a trace.

    `jnp.zeros(())` is the second half and it is not optional: inside a jit even a concrete geometry
    produces TRACERS of that trace, so the apply built from it closes over them. Keeping that closure
    and handing it to a later trace leaks them -- which is what `test_two_separate_traces_do_not_leak`
    is for. The same probe guards `_KRYLOV_CACHE` a few hundred lines down, for the same reason.
    """
    probe = (jnp.zeros(()), fil.pos, fil.mom, fil.length, fil.self_g)
    return not any(isinstance(x, jax.core.Tracer) for x in probe)


def _geom_cached(fil, mu0, what, build):
    """``build()`` once per network for things that depend on GEOMETRY alone, and keep them.

    The partial-inductance apply and the near-field triplets are functions of where the metal is and
    of nothing else -- not the conductivity, not the frequency. They were rebuilt on every
    ``solve()`` regardless: about a quarter of a welded module solve, paid again at every point of a
    frequency sweep and every iteration of a design loop, to produce the same arrays each time.

    A TRACED geometry is never cached -- a bond wire being routed has no fixed value to keep -- so
    the differentiable path is unaffected and keeps rebuilding, which is what it must do.
    """
    if not _frozen_geometry(fil):
        return build()
    key = (id(fil), float(mu0))
    hit = _GEOM_CACHE.get(key)
    if hit is not None and hit[0] is fil:
        if what not in hit[1]:
            hit[1][what] = build()
        return hit[1][what]
    if len(_GEOM_CACHE) >= _GEOM_CACHE_MAX:
        _GEOM_CACHE.pop(next(iter(_GEOM_CACHE)))
    val = build()
    _GEOM_CACHE[key] = (fil, {what: val})
    return val


def solve_network(
    fil: Filaments,
    sigma,
    terminals,
    sources,
    grounds=(),
    currents=(),
    devices=(),
    device_host=None,
    omega=0.0,
    mu0=4e-7 * np.pi,
    matrix_free=None,
    tol=1e-8,
    restart=None,
    mag=None,
    chi=None,
    quad_m=2,
    operator=None,
    magnetic_current=False,
):
    """Solve the PEEC circuit for a network whose terminals are node SETS.

    The general form of :func:`network_impedance`: a terminal is a pad carrying many filament ends,
    held at one potential, and a problem may have several of them. The equations are Ruehli's
    (IEEE Trans. MTT 22(3), 1974, sec. III)::

        Z I - A' phi = 0                       Z = diag(R) + j w Lp,  A = incidence
        phi_n = phi_T      for n in T          a terminal is equipotential
        (A I)_n = 0                            at every node outside a terminal
        phi_A - phi_B = -Z (A I)_A             a device, with (A I)_A + (A I)_B = 0

    and one row per terminal, from the ports: a source fixes ``phi_A - phi_B``, a ground fixes
    ``phi_T``, a current fixes the net current injected there, and a terminal named by nothing at
    all is open -- which is what ``(A I)_T = 0`` already says, so writing ``i(T) - 0`` is a way to
    say out loud what omitting it would mean anyway.

    Args:
        fil: the network.
        sigma: conductivity, scalar or per filament.
        terminals: ``{name: node indices}``, from :func:`terminal_nodes`.
        sources/grounds/currents/devices: as returned by :func:`port_spec`. A device is
            ``(A, B, Z)``: a two-terminal impedance carrying no partial inductance and coupling to
            nothing, which is how a component enters the network without being meshed as metal.
        omega: angular frequency, rad/s.
        mu0: permeability of the surrounding medium.
        mag: the magnetic mesh, on the SAME lattice as ``fil`` (see
            :func:`bar_filaments`'s ``grid_shapes``). ``None`` -- the default and every network
            without a core -- takes a structural branch that leaves the operator exactly as it was.
        chi: susceptibility ``mu_r - 1`` per magnetic element. Complex is allowed, and is core loss.
        quad_m: sub-points per cell axis in the magnetic potential and the coupling.
        magnetic_current: also return the magnetisation current ``I_m``, unscaled, as a fourth
            value -- ``None`` when the model has no core. It is what core loss is computed from,
            and it is otherwise dropped: the solve carries it as the scaled `x_m` and unpacks only
            the electric slices. Gated rather than always returned so the ``cur, phi, inj`` unpack
            every caller uses keeps working.
        operator: ``jno.solve.hierarchical(...)`` to compress the DENSE blocks of a welded network --
            a non-lattice part's own partial inductance and every cross block between parts. ``None``
            keeps the exact path, so this changes no existing answer. A plain lattice ignores it: the
            FFT is exact and already O(N log N), so there is nothing to win.

    With a core the unknowns grow to ``[I_c, phi, I_m]`` and the system is Torchio's dual pair
    (*A PEEC method with magnetic materials*, IEEE TMTT 66(5), 2018)::

        Z I_c + j w K' I_m - A' phi = 0         the magnetisation's flux links the circuit
        (R_m + A_m' P_m A_m) I_m = K I_c        the mmf the current drives around the core

    ``phi_m`` is eliminated rather than carried: the magnetic charge is ``-A_m I_m`` and its
    potential is ``P_m`` times that, so substituting leaves three blocks instead of four and the
    magnetic operator is the exact dual of the electric ``R + j w Lp``.

    Args (continued):
        restart: Krylov subspace size. Defaults by structure -- 16 for a plain lattice, 64 for a
            welded network OR one carrying a magnetic core, both of which need the larger subspace
            and are faster with it (1.5x welded, 3x with a core) while a plain lattice is 2.9x
            slower. 16 is where a LATTICE solve is cheapest, and the ADJOINT uses the
            same value -- it is preconditioned by ``P^T`` rather than by ``P``, so it converges like
            the primal. It did not always: with the forward preconditioner on the transposed
            operator the tangent system looked intractable, and at 592 elements the gradient came
            back with the wrong SIGN after 310x the forward cost. It is now 2.5x and exact.
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

    Asp0 = fil.incidence.tocsr()
    nn, ne = Asp0.shape
    sig = jnp.broadcast_to(jnp.asarray(sigma, dtype=float), (ne,))
    # The element's own impedance is a shape-aware SURFACE one, not rho*l/A. It reduces to the DC
    # value below the skin depth, so nothing changes at low frequency; above it, the current retreats
    # to the surface and a conductor no longer has to be split across its section to say so.
    _check_unresolved_thickness(fil, sig, omega, mu0)
    R, pidx, pair_z = _element_impedance(fil, omega, sig, mu0)
    pidx_j = None if pidx is None else jnp.asarray(pidx)
    lat = getattr(fil, "lattice", None)
    welded = isinstance(lat, dict) and "welded" in lat
    if welded and operator is None and any(b[2] is not None and b[2].get("graded") for b in lat["welded"]):
        raise ValueError(
            "peec.solve_network: a welded part is a GRADED lattice, which is not block-Toeplitz and "
            "has no FFT. Pass `operator=jno.solve.hierarchical(...)`."
        )
    has_lattice = lat is not None and (not welded or any(b[2] is not None for b in lat["welded"]))
    free_form = has_lattice if matrix_free is None else bool(matrix_free)
    if lat is not None and lat.get("graded") and operator is None:
        raise ValueError(
            "peec.solve_network: this lattice is GRADED, which is what makes local refinement "
            "possible -- and it is not block-Toeplitz, so there is no FFT for it. Pass "
            "`operator=jno.solve.hierarchical(...)`, which applies an unstructured operator by "
            "hierarchical compression. The dense path is the other option and costs O(N^2) memory."
        )
    if free_form and lat is None:
        raise ValueError(
            "peec.solve_network: matrix_free=True needs a lattice somewhere in the network, and these "
            "filaments have none. A polyline's filaments are not Toeplitz, so there is no structure to "
            "exploit and the dense path is the honest one."
        )
    # The magnetic system, when there is one. Structural: with `mag=None` not one line below this
    # executes, `nm` stays 0, and every branch guarded by it is not taken -- which is what keeps the
    # assembled operator identical to the one every existing network already solves.
    nm, mmf, flux, mag_apply, mag_diag = 0, None, None, None, None
    if mag is not None:
        if not free_form:
            raise ValueError(
                "peec.solve_network: a magnetic system needs the lattice path. The dense path forms "
                "Lp as a matrix, and the magnetic blocks are applied by FFT on the shared cell grid, "
                "so there is no assembled operator for them to join. Drop matrix_free=False."
            )
        if float(omega) == 0.0:
            raise ValueError(
                "peec.solve_network: a magnetic core at DC. The magnetisation reaches the circuit "
                "only through `j w K'`, which is identically zero at omega = 0 -- so the core would "
                "be solved for and then have no effect, and the port inductance returned would be "
                "the coreless one with nothing to show that the core had been dropped. Solve at a "
                "frequency; the inductance is flat well below the first resonance, so any small one "
                "gives the magnetostatic answer."
            )
        if chi is None:
            raise ValueError("peec.solve_network: mag= was given without chi=, so the core has no material.")
        nm = int(np.asarray(mag.length).size)
        mmf, flux = magnetic_coupling_apply(fil, mag, quad_m)
        mag_apply, mag_diag = magnetic_system_apply(mag, chi, mu0, quad_m)
        # SYMMETRIC SCALING of the magnetic block, and it is not cosmetic. A reluctance is
        # `l / (mu0 chi A)`, so the magnetic diagonal is order 1e8 where the electric one is order
        # 1e-4 -- twelve decades apart in one operator. Left-preconditioned GMRES then converges
        # happily in the PRECONDITIONED norm while the residual this function checks, which is the
        # unpreconditioned one, stays orders of magnitude above the tolerance: measured 2.3e-05
        # against a 1e-6 limit at mu_r = 10, and no restart depth fixed it (16, 64 and 200 all
        # failed identically) because the subspace was never the problem.
        #
        # Solving instead for `x_m = sqrt(|g|) I_m` and scaling the magnetic row by `1/sqrt(|g|)`
        # puts that diagonal at unit modulus, which balances the two norms so that converging in one
        # means converging in the other. The magnetic half of the preconditioner is then a phase.
        mag_s = jnp.sqrt(jnp.abs(mag_diag))
        mag_pre = mag_diag / jnp.abs(mag_diag)

    # Krylov subspace size, by STRUCTURE -- the same reasoning as the preconditioner above, and for
    # the same reason. A welded network's port excitation spreads over far more of the operator's
    # eigenmodes (see `near_block`), so it needs a higher-degree residual polynomial and a bigger
    # subspace pays; a plain lattice converges in about 35 steps and a bigger subspace is just
    # O(m^2) orthogonalisation it never uses. Measured, warm solve:
    #
    #     welded module, 27,533 elements    restart 16  49.5 s    restart 64  32.8 s   1.5x FASTER
    #     plain lattice, 10,048 bars        restart 16   0.39 s   restart 64   1.12 s  2.9x SLOWER
    #
    # so neither value suits both, and picking one globally would have made every lattice 3x slower.
    #
    # A MAGNETIC network goes with the welded case, and for the same reason: the preconditioner
    # neglects the electric-magnetic coupling entirely (see `P_inv`), so the residual polynomial has
    # more to do and the subspace has to be deep enough to hold it. Measured on the square-ring core,
    # mu_r = 2000, refining the core from 2 mm to 1 x 2 x 2 mm (1,044 -> 2,160 magnetic elements):
    #
    #     restart 16   FAILS at 5.08e-06 after 30.5 s
    #     restart 64   converges in  9.5 s      <- and it is 3x FASTER for converging at all
    #     restart 128  converges in 10.8 s
    #
    # Core STRENGTH is not what costs: mu_r = 2000, 20,000 and 200,000 all converge at restart 16 on
    # the coarse mesh in 1.7 s, with L linear in mu_r to four digits. It is the mesh that costs.
    if restart is None:
        restart = 64 if (welded or nm) else 16

    if free_form:
        scale = mu0 / (4.0 * jnp.pi)
        lp_apply = _geom_cached(
            fil,
            mu0,
            # the operator spec is part of the KEY, not just the value: a compressed apply and an
            # exact one are different operators over identical geometry, and a cache that ignored
            # that would hand a `solve(operator=...)` the previous exact apply, or the reverse
            ("lp", None if operator is None else (operator.tol, operator.leaf, operator.eta, operator.floor)),
            lambda: (
                welded_apply(fil, lambda r: 1.0 / r, mu_scale=scale, operator=operator)
                if welded
                else (
                    _hier_apply(fil, lambda r: 1.0 / r, scale, operator)
                    if lat.get("graded")
                    else lattice_apply(fil, lambda r: 1.0 / r, mu_scale=scale)
                )
            ),
        )
        Z = None
    else:
        Lp = pair_matrix(fil.pos, fil.mom, lambda r: 1.0 / r, fil.self_g, group=fil.group) * (mu0 / (4.0 * jnp.pi))
        Z = jnp.diag(R.astype(complex)) + 1j * float(omega) * Lp
        if pidx is not None:  # the other face of the same slab
            Z = Z.at[jnp.arange(R.shape[0]), pidx_j].add(pair_z)

    names = list(terminals)
    # A terminal is `{name: idx}`, or `{name: (idx, weights)}` to make it a prescribed current
    # DISTRIBUTION rather than a short -- which is what lets its position be differentiated.
    wts = {t: v[1] for t, v in terminals.items() if isinstance(v, tuple)}
    idx = {t: np.asarray(v[0] if isinstance(v, tuple) else v, dtype=int) for t, v in terminals.items()}
    for t, w in wts.items():
        if np.shape(w)[0] != idx[t].size:
            raise ValueError(
                f"jno.peec: terminal {t!r} was given {np.shape(w)[0]} weights for {idx[t].size} nodes. "
                "A weighted terminal needs one weight per node of its support."
            )
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
    # A two-terminal element needs TWO rows for its two terminals: its constitutive law at the
    # positive one, and current continuity through it at the other. Grounding the far side -- what a
    # source does -- would be wrong here: a device's terminal potential is set by the rest of the
    # network, not chosen.
    # `zh` is the same impedance for the PRECONDITIONER, which is built on the host. When Z is
    # traced -- an electro-thermal loop re-impressing R_ds(on) -- it has no value inside the trace,
    # so the DECLARED one stands in: a preconditioner only has to accelerate, and the nominal design
    # point is a good approximation of a value that moves a few percent with temperature.
    for a, b_, z in devices:
        zh = _host_z(z, fallback=(device_host or {}).get(a))
        claim(a, ("a device", ("device", a, b_, z, zh)))
        claim(b_, ("the same device's return", ("devret", a, b_, z, zh)))

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

    # A conductor no terminal touches FLOATS: add any constant to its potential and every current is
    # unchanged, so its block is singular by exactly one direction per floating piece. Whether that
    # was fatal depended on which preconditioner the network happened to take -- a plain lattice's
    # diag(Z) Schur complement absorbed it, a WELDED network's whole-system LU raised "Factor is
    # exactly singular" from inside a callback, saying nothing about conductors.
    #
    # Pinning one node of each floating piece removes the null direction and changes NOTHING
    # physical: `1' A = 0` on an isolated component, so the summed current balance is identically
    # zero and the KCL row the pin replaces is implied by the others. Housekeeping, not a modelling
    # choice -- which is why it is done rather than demanded, and only reported.
    #
    # A ground plane under a trace layer is exactly this case, and it is why it matters.
    import scipy.sparse as _sps  # `sp` is rebound locally further down, so it is not usable here
    from scipy.sparse.csgraph import connected_components as _cc

    _gm = (Asp0 @ Asp0.T).tocoo()
    _nc, _comp = _cc(
        _sps.coo_matrix((np.ones(_gm.row.size, dtype=np.int8), (_gm.row, _gm.col)), shape=(nn, nn)),
        directed=False,
    )
    pinned = []
    if _nc > 1 and len(owner):
        _held = set(_comp[np.asarray(sorted(owner), dtype=int)].tolist())
        for _c in range(_nc):
            if _c in _held:
                continue
            _members = np.flatnonzero(_comp == _c)
            _cand = np.intersect1d(_members, free, assume_unique=False)
            if _cand.size:
                pinned.append(int(_cand[0]))
    if pinned:
        free = np.setdiff1d(free, np.asarray(pinned, dtype=int))
        logging.getLogger("jno").info(
            "peec: %d conductor piece(s) carry no terminal, so their potential floats. One node of "
            "each has been pinned as a reference (nodes %s). This removes a singular direction and "
            "changes no current, impedance or loss -- the balance it replaces is implied by the others.",
            len(pinned),
            ", ".join(str(n) for n in pinned[:6]) + (", ..." if len(pinned) > 6 else ""),
        )

    # The constraint block is assembled as TRIPLETS, not rows. It is current balance plus a handful of
    # port conditions, so it carries a few entries per row against ne + nn columns: on a 12k-bar
    # lattice the incidence alone is 24,548 nonzeros in 61,271,808 slots -- 0.04 % -- and holding that
    # densely costs 1.4 GB and an 86-million-op matvec per Krylov step, next to a 3.7 ms FFT apply.
    def _pot(t, sign):
        """(cols, values, host values) for a terminal's potential in a port row.

        Unweighted it is the first node's -- the others are shorted to it, so any of them would do.
        Weighted it is the WEIGHTED AVERAGE over the pad, which is what a terminal that is not a
        short actually sees.
        """
        ids = idx[t]
        w = wts.get(t)
        if w is None:
            c = np.array([ne + int(ids[0])])
            return c, jnp.array([sign], dtype=complex), np.array([sign], dtype=complex)
        n = len(ids)
        wa = jnp.asarray(w).reshape(-1)
        return (
            ne + ids,
            sign * wa.astype(complex),
            sign * np.array([_host_z(wa[j], fallback=1.0 / n) for j in range(n)]),
        )

    ar, ac_ = Asp0.nonzero()
    aval = np.asarray(Asp0[ar, ac_]).reshape(-1)
    rows_of = lambda n: Asp0.indices[Asp0.indptr[n] : Asp0.indptr[n + 1]]
    vals_of = lambda n: Asp0.data[Asp0.indptr[n] : Asp0.indptr[n + 1]]
    # `vv` carries the exact values (a device impedance may be TRACED); `vv_h` is the host copy the
    # near-field preconditioner is built from, which must stay concrete.
    rr, cc, vv, vv_h, rhs = [], [], [], [], [np.zeros(ne, dtype=complex)]
    r0 = 0
    # Current balance away from the terminals: one row per free node, and the row IS that node's
    # row of the incidence. Taken as a slice rather than a loop -- the loop was 40,000 python
    # iterations on a 113,800-bar network, and `solve_network`'s own bytecode was 21 % of the solve.
    for _n in pinned:  # phi = 0 at one node of each floating piece: a reference, nothing more
        rr.append(np.array([r0]))
        cc.append(np.array([ne + int(_n)]))
        vv.append(np.array([1.0]))
        vv_h.append(np.array([1.0]))
        rhs.append(np.zeros(1, dtype=complex))
        r0 += 1
    if free.size:
        sub = Asp0[free]
        counts = np.diff(sub.indptr)
        rr.append(r0 + np.repeat(np.arange(free.size), counts))
        cc.append(sub.indices)
        vv.append(sub.data)
        vv_h.append(sub.data)
        rhs.append(np.zeros(free.size, dtype=complex))
        r0 += int(free.size)
    for t in names:
        ids = idx[t]
        w = wts.get(t)
        if w is None:
            # A terminal is equipotential: tie its nodes to its first. A real pad is a short, and a
            # short is what this is -- but it is also why a pad cannot MOVE: which nodes are in the
            # set is a step function of position, so the answer is piecewise constant in it
            # (measured: 0.000 A for a quarter-millimetre slide, then a 6.5 A jump).
            for n in ids[1:].tolist():
                rr.append(np.array([r0, r0]))
                cc.append(np.array([ne + n, ne + int(ids[0])]))
                vv.append(np.array([1.0, -1.0]))
                vv_h.append(np.array([1.0, -1.0]))
                r0 += 1
                rhs.append(np.zeros(1, dtype=complex))
        else:
            # Weighted: the pad is NOT shorted. Instead the terminal's current enters each node in
            # proportion to its weight -- `w_1 (A I)_i = w_i (A I)_1` for every other node i -- and
            # the port sees the weighted-average potential (see `_pot`). Same row count, same
            # unknowns: the k-1 tie rows become k-1 ratio rows.
            #
            # That is what makes a terminal's POSITION differentiable. The node set is a frozen
            # superset covering the travel; the weights are smooth in the position and carry the
            # gradient, which is the same structure-frozen/values-traced split as everywhere else.
            wa = jnp.asarray(w).reshape(-1)
            nw = len(ids)
            hw = np.array([_host_z(wa[j], fallback=1.0 / nw) for j in range(nw)])
            # Anchor on the LARGEST weight, not the first node. A placement weight is a bump, so on a
            # support wide enough to travel across the far end's weight underflows -- exp(-36) at one
            # end of an 11 mm support -- and every ratio row written against it degenerates to the
            # same statement, leaving the block rank deficient. The rows span the same constraint
            # whichever node anchors them ("the injection is parallel to w"), so this changes the
            # conditioning and nothing else.
            r_i = int(np.argmax(np.abs(hw)))
            n0 = int(ids[r_i])
            k1, v1 = rows_of(n0), vals_of(n0)
            h0 = hw[r_i]
            for j in [q for q in range(nw) if q != r_i]:
                kj, vj = rows_of(int(ids[j])), vals_of(int(ids[j]))
                hj = hw[j]
                # Each ratio row is HOMOGENEOUS, so its scale is free -- and it has to be taken. A
                # smooth placement weight is a bump: over a support wide enough to travel across,
                # its values span ten-plus decades, and rows scaled by them left the constraint
                # block so badly conditioned that the Krylov solve stalled at 1.7e-03. Dividing by
                # the larger of the two weights puts every row at O(1) and changes nothing else.
                sc = jnp.maximum(jnp.abs(wa[r_i]), jnp.abs(wa[j]))
                sc_h = max(abs(h0), abs(hj)) or 1.0
                rr.append(np.full(k1.size + kj.size, r0))
                cc.append(np.concatenate([kj, k1]))
                vv.append(jnp.concatenate([(wa[r_i] / sc) * jnp.asarray(vj), -(wa[j] / sc) * jnp.asarray(v1)]))
                vv_h.append(np.concatenate([(h0 / sc_h) * vj, -(hj / sc_h) * v1]))
                r0 += 1
                rhs.append(np.zeros(1, dtype=complex))
    for t in names:  # and one row for the terminal itself
        kind, *rest = row[t][1] if t in row else ("open", t, 0.0 + 0j)
        if kind == "source":
            a, b_, g = rest
            ca, va, ha = _pot(a, 1.0)
            cb, vb, hb = _pot(b_, -1.0)
            rr.append(np.full(ca.size + cb.size, r0))
            cc.append(np.concatenate([ca, cb]))
            vv.append(jnp.concatenate([va, vb]))
            vv_h.append(np.concatenate([ha, hb]))
        elif kind == "device":  # phi_A - phi_B = Z I_dev,  I_dev = -(A I)_A
            a, b_, z, zh = rest
            g = 0.0 + 0j
            # (A I)_T is the current injected INTO the metal at T, which is why a source's + terminal
            # reads positive. A device is the other way round: it DRAWS its current out of the metal,
            # so the current through it is -(A I)_A and the term enters with a plus. Getting this
            # backwards gives a passive device a negative resistance, which is what the series oracle
            # caught: 2 R_wire - R_dev instead of 2 R_wire + R_dev.
            col = np.asarray(Asp0[idx[a]].sum(0)).reshape(-1)
            k = np.flatnonzero(col)
            ca, va, ha = _pot(a, 1.0)
            cb, vb, hb = _pot(b_, -1.0)
            rr.append(np.full(ca.size + cb.size + k.size, r0))
            cc.append(np.concatenate([ca, cb, k]))
            # jnp, not numpy: Z may be TRACED. A device whose impedance depends on the solved state
            # is the whole electro-thermal feedback -- R_ds(on) rises with the junction it heats.
            vv.append(jnp.concatenate([va, vb, jnp.asarray(z, dtype=complex) * jnp.asarray(col[k])]))
            # The preconditioner's copy. A traced Z has no value inside a jaxpr trace, and the
            # preconditioner only has to accelerate -- so it falls back to an IDEAL device, Z = 0,
            # a short across the terminals. Convergence may suffer; the residual check still guards
            # the answer, and the OPERATOR above carries the exact traced value either way.
            vv_h.append(np.concatenate([ha, hb, zh * col[k]]))
        elif kind == "devret":  # (A I)_A + (A I)_B = 0 -- what goes in comes out
            a, b_, _z, _zh = rest
            g = 0.0 + 0j
            col = np.asarray(Asp0[np.concatenate([idx[a], idx[b_]])].sum(0)).reshape(-1)
            k = np.flatnonzero(col)
            rr.append(np.full(k.size, r0))
            cc.append(k)
            vv.append(col[k])
            vv_h.append(col[k])
        elif kind == "ground":
            _t, g = rest
            ct, vt, ht = _pot(t, 1.0)
            rr.append(np.full(ct.size, r0))
            cc.append(ct)
            vv.append(vt)
            vv_h.append(ht)
        else:  # a fixed injected current, or an open terminal (which is zero injected current)
            _t, g = rest
            col = np.asarray(Asp0[idx[t]].sum(0)).reshape(-1)
            k = np.flatnonzero(col)
            rr.append(np.full(k.size, r0))
            cc.append(k)
            vv.append(col[k])
            vv_h.append(col[k])
        rhs.append(np.atleast_1d(np.asarray(g, dtype=complex)))
        r0 += 1
    if r0 != nn:
        raise ValueError(f"jno.peec: built {r0} constraint rows, expected {nn}; this is a bug.")
    crow, ccol = np.concatenate(rr), np.concatenate(cc)
    cval_h = np.concatenate(vv_h).astype(complex)
    # The sparsity STRUCTURE is never traced, but a VALUE may be: a device impedance that depends on
    # the solved state is what an electro-thermal fixed point re-impresses every pass (a SiC die's
    # R_ds(on) rises ~0.5 %/K). So the values go through jnp and only the pattern stays numpy.
    cval = _fuse_triplets(vv)
    b = jnp.asarray(np.concatenate(rhs))
    if nm:  # the magnetic rows are homogeneous: the core is driven by the current, not by a port
        b = jnp.concatenate([b, jnp.zeros(nm, b.dtype)])
    _refuse_disconnected(Asp0, idx, sources, grounds, currents, devices)

    if free_form:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        w = float(omega)
        Rc = R.astype(complex)
        # a dummy when nothing is paired, so `_run`'s arity does not depend on the network
        czc = pair_z if pidx is not None else jnp.zeros(1, dtype=complex)
        zdiag = Rc + (1j * w) * jnp.asarray(_lattice_diag(fil, mu0))

        # The preconditioner is built on the host from concrete numbers -- it only has to accelerate,
        # and no gradient runs through it. Under jax.grad a traced value still has a concrete primal;
        # inside a jit it has none, and that is worth saying plainly rather than failing in scipy.
        C = sp.coo_matrix((cval_h, (crow, ccol)), shape=(nn, ne + nn)).tocsr()
        CI, Cp = C[:, :ne], C[:, ne:]
        Cj = _bcoo(cval, crow, ccol, (nn, ne + nn))
        # The incidence transpose is sparse too — two entries per column, since a filament has two
        # ends. Left dense it is an ne x nn complex matrix hit twice per Krylov step: at 6,806 bars
        # that is 303 MB and about 55 ms a step, against a 5.9 ms FFT apply, so it was ten times the
        # operator it was helping to apply.
        Acj = _bcoo(aval.astype(complex), ac_, ar, (ne, nn))

        # The preconditioner is a sparse LU of the WHOLE block system, with Z replaced by its NEAR
        # FIELD -- the diagonal plus every pair within a couple of element sizes. Not a Schur
        # complement of the diagonal, which is what this used to be:
        #
        #     welded network, elements    354    1002    1552
        #     diag(Z) Schur, iterations   106     865    3441
        #     Z_near whole-system LU       11      16      30
        #
        # A diagonal Z leaves the spectrum spread enough that the port excitation of a welded network
        # -- which excites far more eigenmodes than a plain lattice's, see near_block -- costs a
        # high-degree residual polynomial. The near field compresses it until that stops mattering.
        #
        # The whole system stays sparse (about 30 nonzeros a row) and so does its factorisation (60 to
        # 120), so this is one sparse LU rather than a dense Schur complement, and it replaces both
        # halves of the old preconditioner.
        # A WELDED network gets the near field; a plain lattice does not. Measured, iterations to a
        # 1e-8 relative residual:
        #
        #     welded elements          354   1002   1552        lattice, 1584 elements
        #     diag(Z) Schur            106    865   3441        converges
        #     Z_near whole-system LU    11     16     30        does NOT converge
        #
        # Each suits its own case and neither suits both, so the choice is made by structure rather
        # than by picking a winner. Why a welded network needs it at all is recorded on near_block:
        # its port excitation spreads over far more eigenmodes, and only a compressed spectrum makes
        # the residual polynomial cheap enough.
        nr, nc, nv = (
            _geom_cached(fil, mu0, "near", lambda: near_block(fil, lambda r: 1.0 / r, mu_scale=mu0 / (4.0 * jnp.pi)))
            if welded
            else (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
        )
        AT = Asp0.T.tocsr()
        # the dtype follows jax_enable_x64 rather than being pinned
        cdtype = jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64
        shape_out = jax.ShapeDtypeStruct(((ne + nn) if welded else nn,), cdtype)
        el = ccol < ne
        CIj = _bcoo(cval[el], crow[el], ccol[el], (nn, ne))
        # the same two blocks the other way round, for the transposed preconditioner
        CIjT = _bcoo(cval[el], ccol[el], crow[el], (ne, nn))
        Aj = _bcoo(aval.astype(complex), ar, ac_, (nn, ne))

        # Keyed by the network's IDENTITY, and the network is kept alive in the entry: an id alone is
        # reusable after a garbage collection, and a stale hit would silently run the previous
        # geometry's closures.
        #
        # And NOT reused across traces. A closure built inside a jit is only valid inside that jit:
        # even a bare `jnp.zeros(())` made there binds to the trace, so a later hit raises
        # UnexpectedTracerError. That never fired while every call rediscretised -- a fresh `fil`
        # missed the cache every time -- and appeared the moment `.build()` made the identity stable.
        # Skipping it under a trace costs nothing: XLA caches the compiled executable, so the reuse
        # this was written for (an eager sweep over frequencies) is the only case it was ever paying.
        # ...and keyed on the CONSTRAINT VALUES too, because the compiled closure captures the
        # constraint matrix. A device impedance lives in there, so re-solving the same network with a
        # different Z -- exactly what an electro-thermal loop does every pass -- returned the FIRST
        # call's operator. It showed up as a solve that would not converge (7.5e-02) rather than as a
        # wrong answer, but only because the residual check was there to catch it. Like the trace
        # guard below, this was invisible while every call rediscretised and a fresh network missed
        # the cache anyway.
        # "Are we inside ANY transform?" -- and a bare `jnp.zeros(())` probe answers only half of it.
        # Under jit everything stages, so the probe is a tracer and the guard fires. Under jax.grad
        # nothing stages except what DEPENDS on the differentiated input, so the probe stays concrete
        # while `R` (from a traced conductivity) and `cval` (from a traced device impedance) do not.
        # Caching then stored a closure holding LinearizeTracers, and the next eager call reusing it
        # raised UnexpectedTracerError far from here. So the values the closure can capture are what
        # gets checked, not a probe.
        # `chi` joins the probe for the same reason `cval` and `R` are in it: the compiled closure
        # captures the magnetic operator, which is built from chi. Under `jax.grad` through a
        # permeability nothing else here stages, so without chi the guard would let a closure holding
        # LinearizeTracers into the cache and the next eager call would raise far from here.
        traced = any(
            isinstance(x, jax.core.Tracer)
            for x in (jnp.zeros(()), cval, R, fil.pos, fil.length, fil.area, *(() if chi is None else (chi,)))
        )
        # ...and the key carries the core, so re-solving one network against a different permeability
        # cannot reuse the first call's operator. Same failure shape as the device impedance above.
        mag_tag = 0 if (traced or not nm) else hash(np.asarray(chi).tobytes())
        key = (
            id(fil),
            ne,
            nn,
            float(tol),
            int(restart),
            hash(cval_h.tobytes()),
            nm,
            mag_tag,
            None if operator is None else (operator.tol, operator.leaf, operator.eta, operator.floor),
        )
        entry = None if traced else _KRYLOV_CACHE.get(key)
        cached = entry[1] if (entry is not None and entry[0] is fil) else None

        # What the factorisation is OF, hashed once: the constraint blocks and the incidence are
        # fixed inside this closure, so the only things that can change between calls are `zd` (the
        # conductivity, through R) and the frequency. Cheap to compute here, and it makes the cache
        # key below exact rather than an identity guess -- `id(CI)` is reusable after a collection,
        # and a stale hit would silently precondition with the previous network's matrices.
        _fact_static = (
            ne,
            nn,
            hash(CI.data.tobytes()),
            hash(Cp.data.tobytes()),
            hash(AT.data.tobytes()),
            hash(np.asarray(nv).tobytes()),
        )

        def _factorise(zd, w_now):
            """Build and stash the preconditioner's factorisation for the CURRENT frequency.

            Through a callback rather than inline, because ``zd`` depends on the conductivity and is
            therefore a TRACER under ``jax.grad``. Nothing differentiable passes through here: a
            preconditioner only has to accelerate, and the gradient comes from the outer
            ``custom_linear_solve``. The frequency is an ARGUMENT because one compiled solve serves a
            whole sweep, so a value closed over here would be the first point's for every later one.
            """
            zdn = np.asarray(zd)
            # The sparse LU is the single most expensive thing in the solve once the mesh is real:
            # profiled on GPU at 40,000 nodes it was 1.515 s per solve, 53 % of the whole thing, and
            # its fill-in (21x the nonzeros at 8,640 nodes, worse above) grows faster than the device
            # work it is helping. Re-running it for values it has already seen buys nothing.
            #
            # Keyed on CONTENT, like `_FACTOR_CACHE` in utils/solver/linear.py, and holding exactly
            # ONE entry -- the same one `_LU_HOLDER` was already keeping alive, so this costs no
            # memory. An LU of a 40,000-node Schur complement is hundreds of megabytes; a cache that
            # kept several would trade a solve's time for a machine's.
            #
            # What this does NOT do is reuse a factorisation across a CHANGING conductivity, which is
            # what a design loop does every iteration. That would be a different claim -- that a
            # stale preconditioner still accelerates -- and it is not made here.
            ckey = (_fact_static, float(w_now), hash(zdn.tobytes()))
            if _LU_HOLDER.get("key") == ckey and _LU_HOLDER.get("lu") is not None:
                return np.zeros((), dtype=np.float64)
            _LU_HOLDER["key"] = None  # a failed factorisation must not leave a key claiming success
            if nv.size:  # the near field, for a welded network
                off = (1j * float(w_now)) * nv
                zblk = sp.coo_matrix(
                    (
                        np.concatenate([zdn, off, off]),
                        (np.concatenate([np.arange(ne), nr, nc]), np.concatenate([np.arange(ne), nc, nr])),
                    ),
                    shape=(ne, ne),
                ).tocsr()
                _LU_HOLDER["lu"] = spla.splu(sp.bmat([[zblk, -AT], [CI, Cp]], format="csc"))
                _LU_HOLDER["kind"] = "whole"
            else:  # a plain lattice: the Schur complement of the diagonal, which is what suits it
                _LU_HOLDER["lu"] = spla.splu((CI @ sp.diags(1.0 / zdn) @ AT + Cp).tocsc())
                _LU_HOLDER["kind"] = "schur"
            _LU_HOLDER["key"] = ckey
            return np.zeros((), dtype=np.float64)

        @jax.custom_jvp
        def _token(zd, w_now):
            return jax.pure_callback(_factorise, jax.ShapeDtypeStruct((), jnp.float64), zd, w_now)

        @_token.defjvp
        def _token_jvp(primals, tangents):
            return _token(*primals), jnp.zeros(())  # a constant token; the refusal is on _precond

        def _precond(r):
            return jax.pure_callback(_holder_solve, shape_out, r)

        def _precond_t(r):
            return jax.pure_callback(_holder_solve_t, shape_out, r)

        def M_apply(x, w_, zd, Rc_, cz_):
            cur, phi = x[:ne], x[ne : ne + nn]
            zi = Rc_ * cur
            if pidx_j is not None:  # structural, so this branch is taken once at trace time
                zi = zi + cz_ * cur[pidx_j]  # the sheet on the slab's other face
            top = zi + (1j * w_) * lp_apply(cur) - Acj @ phi
            if not nm:
                return jnp.concatenate([top, Cj @ x])
            # The magnetisation links flux into the circuit (top) and the circuit drives an mmf
            # around the core (bottom). ONE block, applied both ways round -- see
            # `magnetic_coupling_apply` -- so the coupled operator is symmetric by construction
            # rather than by two matrices that have to be kept in agreement.
            im = x[ne + nn :] / mag_s  # the unknown is the SCALED magnetisation; see mag_s above
            return jnp.concatenate([top + (1j * w_) * flux(im), Cj @ x[: ne + nn], (mag_apply(im) - mmf(cur)) / mag_s])

        def _elec_pre(r, zd):
            if welded:
                return _precond(r)  # one factorisation of the whole block system
            ri, rp = r[:ne], r[ne:]  # the Schur split, for a plain lattice
            dphi = _precond(rp - CIj @ (ri / zd))
            return jnp.concatenate([(ri + Acj @ dphi) / zd, dphi])

        def P_inv(r, zd):
            if not nm:
                return _elec_pre(r, zd)
            # The coupling is NEGLECTED in the preconditioner and the two halves are preconditioned
            # independently -- pypeec's choice, and it is what keeps the electric preconditioner the
            # one that is already measured and tuned rather than a new one built around a core.
            # The magnetic half takes its own diagonal, which for `R_m + A' P_m A` is exact.
            return jnp.concatenate([_elec_pre(r[: ne + nn], zd), r[ne + nn :] / mag_pre])

        def _elec_pre_t(r, zd):
            """``P^-T``, written out rather than obtained by transposing ``P_inv`` automatically.

            The host factorisation is a callback, which jax cannot transpose on its own, so the block
            algebra is done here: with ``D = diag(zd)``, ``P_inv`` is

                [[D^-1 - D^-1 A^T S^-1 C_I D^-1,  D^-1 A^T S^-1],
                 [        -S^-1 C_I D^-1       ,      S^-1     ]]

            and this is its transpose, with ``S^-T`` supplied by the same factorisation applied the
            other way round.
            """
            if welded:
                return _precond_t(r)
            ri, rp = r[:ne], r[ne:]
            u = _precond_t(Aj @ (ri / zd) + rp)
            return jnp.concatenate([(ri - CIjT @ u) / zd, u])

        def P_inv_T(r, zd):
            # a diagonal is its own transpose, so only the electric half needs the block algebra
            if not nm:
                return _elec_pre_t(r, zd)
            return jnp.concatenate([_elec_pre_t(r[: ne + nn], zd), r[ne + nn :] / mag_pre])

        # A SHORT restart. The block preconditioner is strong -- measured against an EXACT Z^-1
        # preconditioner, which converges in one iteration, diag(Z) takes two -- so the solve needs a
        # couple of cycles and jax builds the whole basis whether or not it is needed. Per swept point
        # at 6,806 bars: restart 30 -> 0.342 s, 16 -> 0.308 s, 10 -> 0.308 s, 6 -> 0.386 s,
        # 4 -> 0.754 s, all to the same impedance. Below ten the cycles outnumber what they save. The
        # ADJOINT uses the same value, which it could not before: it was the wrong preconditioner that
        # made the tangent system hard, not the subspace size.
        _RS = min(int(restart), ne + nn + nm)

        if cached is None:

            @jax.jit
            def _run(rhs, w_, zd, Rc_, cz_):
                # sequenced through the right-hand side so the factorisation is built before the
                # solve that reads it, rather than relying on callback ordering
                rhs = rhs + _token(zd, w_).astype(rhs.dtype)

                # OUR OWN custom_linear_solve, rather than the one inside jax's gmres. That one
                # solves the adjoint with the FORWARD preconditioner, which is not a preconditioner
                # for the transposed operator at all -- so the tangent system looked far harder than
                # the primal and the gradient came back wrong without any of it being reported. Here
                # the adjoint gets P^T, and converges like the primal does.
                def _gm(P):
                    return lambda op, r: jax.scipy.sparse.linalg.gmres(
                        op, r, M=P, tol=float(tol), atol=0.0, restart=_RS, maxiter=400
                    )[0]

                return jax.lax.custom_linear_solve(
                    lambda v: M_apply(v, w_, zd, Rc_, cz_),
                    rhs,
                    _gm(lambda v: P_inv(v, zd)),
                    _gm(lambda v: P_inv_T(v, zd)),
                )

            if not traced:
                if len(_KRYLOV_CACHE) > 8:  # bounded: holds compiled code and a reference to a network
                    _KRYLOV_CACHE.pop(next(iter(_KRYLOV_CACHE)))
                _KRYLOV_CACHE[key] = (fil, _run)
        else:
            _run = cached

        x = _run(b, jnp.asarray(w), zdiag, Rc, czc)
        resid = jnp.linalg.norm(M_apply(x, w, zdiag, Rc, czc) - b) / jnp.maximum(jnp.linalg.norm(b), 1e-300)
        _check(resid, _converged, max(1e2 * float(tol), 1e-9))
    else:
        top = jnp.concatenate([Z, -jnp.asarray(Asp0.toarray()).T.astype(complex)], axis=1)
        bot = jnp.zeros((nn, ne + nn), dtype=complex).at[crow, ccol].add(jnp.asarray(cval))
        M = jnp.concatenate([top, bot], axis=0)
        x = jnp.linalg.solve(M, b)
    _check(jnp.all(jnp.isfinite(x)), _finite)
    cur, phi = x[:ne], x[ne : ne + nn]
    inj = {t: jnp.asarray(np.asarray(Asp0[idx[t]].sum(0)).reshape(-1), dtype=complex) @ cur for t in names}
    if magnetic_current:
        # UNSCALED. The unknown solved for is `x_m = sqrt(|g|) I_m` (see the symmetric scaling
        # above), so handing back the raw slice would be the magnetisation current times a
        # twelve-decade factor -- right shape, wrong number, and nothing downstream could tell.
        return cur, phi, inj, (None if mag is None else x[ne + nn :] / mag_s)
    return cur, phi, inj


def _skin_depth(sigma, freq, mu0=4e-7 * np.pi):
    """The skin depth a discretisation should be gated on, or None when it cannot be read.

    The GATE only, never a solved quantity: it decides structure, which must be fixed for the built
    network, so a swept solve takes the highest frequency -- the conservative case, since that is
    where a conductor is thickest against the skin depth. Unreadable conductivity (a traced design
    variable with no value yet) means no pairing, which is the old model and never worse than it.
    """
    try:
        f = float(np.max(np.asarray(freq, dtype=float)))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f) or f <= 0.0 or sigma is None:
        return None
    try:
        s = np.asarray(jax.lax.stop_gradient(jnp.asarray(sigma)), dtype=float)
    except (TypeError, ValueError, jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return None
    s = s[np.isfinite(s) & (s > 0)]
    if s.size == 0:
        return None
    return float(np.sqrt(2.0 / (2.0 * np.pi * f * mu0 * float(np.max(s)))))


def _sheet_families(axis, skin, span, d, delta):
    """Which axis families should carry a current sheet per FACE, and the thickness between them.

    Only a family that is entirely one conductor thickness can be split: the two sheets are a second
    lattice family offset from the first, and a lattice family has ONE sub-point offset or it stops
    being Toeplitz (see :func:`~jno.utils.solver.kernel.lattice_kernel`). A layout mixing thicknesses
    on one grid -- traces over a plane -- therefore keeps the single-current model for now.

    Returns ``{axis: (thin_axis, thickness)}``, empty when nothing qualifies, which is every network
    holding no conductor thick against the skin depth.
    """
    out = {}
    if delta is None or not np.isfinite(delta) or delta <= 0:
        return out
    for ax in sorted(set(np.asarray(axis).tolist())):
        sel = np.asarray(axis) == ax
        if not (np.asarray(span)[sel] == 1).all():
            continue  # already split across its thickness: the elements resolve it themselves
        th = np.unique(np.round(np.asarray(skin)[sel], 12))
        if th.size != 1:
            continue  # mixed thicknesses cannot share one offset
        thick = float(th[0])
        if thick <= 2.0 * delta:
            continue  # the current fills the section, and one element already says so exactly
        # span == 1, so the conductor IS one cell thick and the thin axis is the matching pitch
        cand = [c for c in range(3) if c != ax and abs(float(d[c]) - thick) < 1e-12]
        if len(cand) == 1:
            out[int(ax)] = (int(cand[0]), thick)
    return out


def _element_impedance(fil, omega, sig, mu0):
    """Per-element impedance, and the coupling where a slab carries a current sheet on each face.

    Returns ``(R, pair_idx, pair_z)``, with ``pair_idx`` None when nothing is paired -- which is
    every network holding no conductor thick against the skin depth, so the common case pays nothing
    at all, not even a gather.

    A paired element takes the DIAGONAL of the 2-port slab impedance rather than the one-unknown
    form, and its partner supplies the off-diagonal. The two agree by construction where it matters:
    equal sheet currents give back exactly what :func:`internal_impedance` would have returned (see
    :func:`slab_transfer_impedance`), so switching a conductor to a pair cannot move an answer that
    was already right.
    """
    R = internal_impedance(fil.length, fil.area, fil.skin, fil.round_, omega, sig, mu0, fil.span)
    idx = getattr(fil, "pair", None)
    if idx is None:
        return R, None, None
    idx = np.asarray(idx, dtype=int)
    live = idx >= 0
    if not live.any():
        return R, None, None
    # a paired element is a slab by construction, so area / thickness is its in-plane width
    z_self, z_mut = slab_transfer_impedance(fil.length, fil.area / fil.skin, fil.skin, omega, sig, mu0)
    keep = jnp.asarray(live)
    return (
        jnp.where(keep, z_self, R.astype(complex)),
        np.where(live, idx, np.arange(idx.size)),
        jnp.where(keep, z_mut, 0.0 + 0j),
    )


def _check_unresolved_thickness(fil, sigma, omega, mu0):
    """Flag a conductor split across its thickness by cells too coarse to resolve the skin depth.

    Such a discretisation satisfies NEITHER model. The surface impedance needs the element to be the
    whole thickness, so a split conductor cannot use it (see :func:`internal_impedance`); and the
    field solve only shows the skin effect if the cells resolve it, which coarse ones do not. What
    comes out is the DC resistance wearing a high-frequency label -- on a 40 x 4 x 2 mm bar at
    1 MHz, 82 uOhm against the 1239 uOhm the one-cell model gives.

    Loud either way, but only fatal when it is the MODEL rather than a detail of it. Real geometry
    has local thick spots -- a terminal post standing on a trace is a 1.57 mm column of metal where
    the trace it sits on is 0.57 mm -- and refusing a whole package because 180 elements of 10,994
    are one would make ordinary models unsolvable while barely moving the port impedance. So a
    minority warns and a majority raises; the inductance, which does not depend on this at all, is
    the common reason to want the minority case anyway.
    """
    w = float(omega)
    if w == 0.0:
        return  # at DC every element is rho l / A and the thickness does not matter
    span = np.asarray(fil.span)
    sub = span > 1
    if not sub.any():
        return
    try:
        sig = np.broadcast_to(np.asarray(jax.lax.stop_gradient(jnp.asarray(sigma)), dtype=float), span.shape)
    except (jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        # Inside a jit there is no value to read. Rather than go silent -- a guard that stops
        # guarding the moment you make the loop fast is worse than none -- `peec.build()` runs this
        # ONCE at the declared conductivity, and that is the conservative case: the verdict turns on
        # the skin depth, a design variable only ever lowers sigma, and a lower sigma is a DEEPER
        # skin depth and so a milder verdict. Whatever a density field does to it later, the
        # declared value already answered the worst case.
        return
    delta = np.sqrt(2.0 / (w * mu0 * np.maximum(sig, 1e-300)))
    # non-differentiably: a traced wire GAUGE reaches here once radius is a design variable,
    # and this guard only decides whether to complain -- no gradient runs through it
    thick = np.asarray(jax.lax.stop_gradient(jnp.asarray(fil.skin)))
    cell = thick / np.maximum(span, 1)
    # Two conditions, and both are needed. The cells must be too coarse to resolve the distribution
    # (a cell wider than half a skin depth), AND the skin effect must actually be worth something at
    # this thickness -- below about two skin depths the surface impedance IS the DC value, so a
    # subdivided conductor loses nothing and there is nothing to say.
    bad = sub & (cell > 0.5 * delta) & (thick > 2.0 * delta)
    if not bad.any():
        return
    k = int(np.flatnonzero(bad)[0])
    n = int(bad.sum())
    msg = (
        f"jno.peec: {n} of {bad.size} elements sit in a conductor that is {int(span[k])} elements "
        f"thick where each is {float(cell[k]) * 1e3:.4g} mm, against a skin depth of "
        f"{float(delta[k]) * 1e3:.4g} mm at {w / (2 * np.pi):.4g} Hz -- {float(thick[k]) / float(delta[k]):.1f} "
        "skin depths through it. Neither model applies there: an element may only take the surface "
        "impedance when it IS the whole thickness, and cells this coarse cannot resolve the current "
        "distribution either, so those elements fall back to the DC resistance. Use ONE cell through "
        "the thickness -- exact at any frequency, and a conductor this thick then carries a current "
        "sheet per face -- or at least "
        f"{float(cell[k]) / (0.5 * float(delta[k])):.0f}x finer so two cells fit in a skin depth. "
        "If one cell is what you ASKED for, a thinner conductor sharing this lattice has overridden "
        "it: solids share one pitch, in z as well as in plane, and the finest wins. Give the thick "
        "conductor its own discretisation, or match the pitches."
    )
    if n * 2 > bad.size:
        # A majority is the MODEL being wrong, so it is refused every time it is asked.
        raise ValueError(msg + " Most of this model is in that state, so it is refused rather than returned.")
    # A minority is a detail, and a detail said once is a warning while a detail said a hundred times
    # is noise: a design loop re-solves the same network every iteration and nothing about the
    # geometry it is complaining about changes between them.
    key = (id(fil), round(w, 6), n)
    if key in _THICKNESS_WARNED:
        return
    if len(_THICKNESS_WARNED) > 64:
        _THICKNESS_WARNED.clear()
    _THICKNESS_WARNED.add(key)
    logging.getLogger(__name__).warning(
        "%s The resistance is understated by that much of the model; the inductance is unaffected.", msg
    )


def _fuse_triplets(parts):
    """Concatenate a mixed numpy/jax list with one jax op per RUN of concrete pieces, not per piece.

    ``vv`` carries one entry per constraint ROW, and current balance writes a row per free node --
    thousands on any real lattice. Sending each through ``jnp.asarray(v).reshape(-1)`` is two eager
    dispatches apiece, and on a GPU two host-to-device transfers as well.

    Measured at 8,640 nodes, three warm solves: 26,337 ``jnp.asarray`` calls, 52,887 lax binds and
    28,326 eager primitive dispatches -- about 90 % of the solve, spent assembling a vector rather
    than solving with it. It is also the reason the GPU was no faster than the CPU (0.801 s against
    0.781 s): Python dispatch does not care which device it is dispatching to, and the FFT apply it
    was hiding is 3.2x quicker there.

    Almost every row is concrete -- only a traced device impedance or a placement weight is not --
    so the concrete runs are fused in numpy and only the genuinely traced pieces stay jax. A network
    with no traced values reduces to a single ``jnp.asarray`` of one numpy array.
    """
    out, buf = [], []
    for piece in parts:
        # numpy ONLY: a concrete jax array would have to come back off the device to join a numpy
        # concatenate, which is the transfer this exists to avoid.
        if isinstance(piece, np.ndarray):
            buf.append(np.asarray(piece, dtype=complex).reshape(-1))
            continue
        if buf:
            out.append(jnp.asarray(np.concatenate(buf)))
            buf = []
        out.append(jnp.asarray(piece, dtype=complex).reshape(-1))
    if buf:
        out.append(jnp.asarray(np.concatenate(buf)))
    if not out:
        return jnp.zeros(0, dtype=complex)
    return out[0] if len(out) == 1 else jnp.concatenate(out)


def _refuse_disconnected(A, idx, sources, grounds, currents, devices=()):
    """A port pair with no metal between them is a modelling error, not an infinite impedance.

    ``jnp.linalg.solve`` on a singular system returns ``inf``/``nan`` without complaining, and an
    infinite resistance reads like a physical answer. Two conductors that were meant to touch but do
    not is the common way to get here, so the check names the terminals rather than the matrix.
    """
    # This is a connected-components question, so it is asked as one. It used to be a python
    # union-find over every filament and every node it touches: 685,194 calls to `find` on a
    # 113,800-bar network, 0.186 s a solve and 14 % of it on a GPU, to answer a question about
    # geometry that `build()` had already frozen. `connected_components` does the same in C.
    from scipy.sparse.csgraph import connected_components

    nn, _ne = A.shape
    # A is nodes x filaments with two entries a column -- a filament's two ends -- so `A @ A.T` has
    # an entry wherever one filament touches two nodes, which is exactly the union the loop did.
    g = (A @ A.T).tocoo()

    # Before the terminal ties are added, so it sees the METAL as the geometry left it: a tag is a
    # function of position, so `x < 3*mm` selects that column of the WHOLE model -- every layer of a
    # stack, not only the one the pad sits on. Wiring a port into a ground plane that way is silent
    # and gives plausible wrong answers: on a real DBC module a DC+ tag with no z filter returned
    # 13.6 nH and 621 uOhm where the same model filtered to the trace layer gave 20.8 nH and
    # 2180 uOhm, against a 21.7 nH reference.
    #
    # A warning, not a refusal: two pieces of metal that a bond wire joins later is a legitimate
    # model, and so is a pad deliberately shorting two conductors.
    _n_metal, _metal = connected_components(
        sp.coo_matrix((np.ones(g.row.size, dtype=np.int8), (g.row, g.col)), shape=(nn, nn)), directed=False
    )
    if _n_metal > 1:
        for _t, _ids in idx.items():
            _ids = np.asarray(_ids, dtype=int)
            _spans = sorted(set(_metal[_ids].tolist()))
            if len(_spans) > 1:
                _warn_once(
                    f"peec: terminal {_t!r} covers {len(_ids)} nodes lying in {len(_spans)} pieces of "
                    "metal that are not connected to each other, so it SHORTS them. That is usually a "
                    "coordinate tag with no z filter picking up a second layer -- a ground plane under "
                    "a trace, say -- which wires the port into it and quietly changes both the "
                    "resistance and the loop inductance. If the short is deliberate, ignore this; "
                    "otherwise restrict the tag in z."
                )

    rows, cols = [g.row], [g.col]
    for ids in idx.values():  # a terminal is one electrical point
        ids = np.asarray(ids, dtype=int)
        if ids.size > 1:
            rows.append(np.full(ids.size - 1, ids[0]))
            cols.append(ids[1:])
    for a, b, _z in devices:  # and a device conducts, so it joins the two terminals it sits between
        rows.append(np.array([int(idx[a][0])]))
        cols.append(np.array([int(idx[b][0])]))
    r, c = np.concatenate(rows), np.concatenate(cols)
    adj = sp.coo_matrix((np.ones(r.size, dtype=np.int8), (r, c)), shape=(nn, nn))
    _ncomp, label = connected_components(adj, directed=False)
    for a, b, _g in sources:
        if label[int(idx[a][0])] != label[int(idx[b][0])]:
            raise ValueError(
                f"jno.peec: no conducting path between terminals {a!r} and {b!r}, so no current can "
                "flow and the impedance is not finite. Conductors are joined where the geometry says "
                "the metal touches — check that the parts meant to be in contact actually overlap."
            )


def _occupied_runs(lab, ax):
    """Length of the contiguous SAME-MATERIAL run containing each cell, along ``ax``.

    How many cells one conductor is divided into through a given direction, which decides whether an
    element may take a surface impedance (see :func:`internal_impedance`). Cells carry a material
    label and ``-1`` where there is no metal; a run breaks at a change of material as well as at a
    void. It has to: a die sitting on a trace is 2 cells of METAL but 1 cell of each conductor, and
    the skin effect does not run across the junction between them. Two stacked copper shapes, on the
    other hand, share a label and stay one run -- a terminal post on a trace really is 1.57 mm of
    continuous copper.
    """
    o = np.moveaxis(np.asarray(lab), ax, -1)
    live = o >= 0
    same = np.zeros(o.shape, bool)
    same[..., 1:] = live[..., 1:] & live[..., :-1] & (o[..., 1:] == o[..., :-1])
    fwd, bwd = np.zeros(o.shape, int), np.zeros(o.shape, int)
    acc = np.zeros(o.shape[:-1], int)
    for k in range(o.shape[-1]):
        acc = np.where(live[..., k], np.where(same[..., k], acc + 1, 1), 0)
        fwd[..., k] = acc
    acc = np.zeros(o.shape[:-1], int)
    for k in range(o.shape[-1] - 1, -1, -1):
        nxt = same[..., k + 1] if k + 1 < o.shape[-1] else np.zeros(o.shape[:-1], bool)
        acc = np.where(live[..., k], np.where(nxt, acc + 1, 1), 0)
        bwd[..., k] = acc
    return np.moveaxis(np.where(live, fwd + bwd - 1, 0), -1, ax)


#: Pitch-unification warnings already issued, so a swept solve says it once rather than per point.
_PITCH_WARNED: set = set()


def _warn_once(msg: str) -> None:
    if msg not in _PITCH_WARNED:
        _PITCH_WARNED.add(msg)
        logging.getLogger("jno").warning(msg)


@dataclass(frozen=True)
class GradedGrid:
    """Ask `bar_filaments` for a grid refined toward every conductor face.

    Built by :func:`jno.peec.graded`. Carries only the two numbers the user chooses -- how fine to
    get and how far that reaches -- because everything else (the extent, the base pitch, which
    coordinates are features) is already in the geometry.
    """

    fine: float
    halo: float


def _graded_edges(shapes, lo, hi, base, fine, halo):
    """Cell boundaries at `base` spacing, refined to `fine` within `halo` of any conductor face.

    The features worth resolving are the conductor FACES: current crowds at a trace edge, and every
    real layout here puts a 1.0 mm trace on a plate tens of millimetres across, so no single pitch
    serves both. Each shape contributes the six faces of its own bounding box.

    Boundaries that differ by less than a twentieth of `fine` are MERGED. Unioning coordinate sets
    is the natural way to grade a grid and `np.unique` will not do it: two `arange` calls that both
    contain the same face disagree in the last bits, and the union then holds a cell 1e-19 wide,
    with no volume and a singular operator. `bar_filaments` refuses such a cell by name; this makes
    sure it never has to.
    """
    faces = [[], [], []]
    for sh in shapes:
        bnd = np.asarray(sh.bounds(), dtype=float).reshape(2, -1)
        for a in range(bnd.shape[1]):
            faces[a] += [float(bnd[0, a]), float(bnd[1, a])]
    out = []
    for a in range(3):
        b = float(np.atleast_1d(base)[a] if np.ndim(base) else base)
        if not (hi[a] > lo[a]):  # a flat axis is one cell thick, exactly as the uniform path has it
            out.append(np.array([lo[a], lo[a] + (b if b > 0 else 1.0)]))
            continue
        sets = [np.arange(lo[a], hi[a] + 0.5 * b, b), np.array([lo[a], hi[a]])]
        for e in faces[a]:
            if lo[a] < e < hi[a]:
                sets.append(np.arange(max(lo[a], e - halo), min(hi[a], e + halo) + 0.5 * fine, fine))
        v = np.sort(np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in sets]))
        v = v[(v >= lo[a]) & (v <= hi[a])]
        keep = np.concatenate([[True], np.diff(v) > 0.05 * fine])
        v = v[keep]
        # the ends are the grid, not a suggestion: a merge must never move them
        v[0], v[-1] = lo[a], hi[a]
        out.append(v)
    return out


def _volume_rule(ln, wt, axis, quad: int, quad_t: int):
    """Tensor-product Gauss points over each element's VOLUME.

    Returns ``(offsets, weights)`` shaped ``(n_elem, quad * quad_t**2, 3)`` and
    ``(n_elem, quad * quad_t**2)``. The offsets are relative to the cell centre; the weights are the
    element's current moment split over them, so they sum to its LENGTH -- which is the invariant the
    partial-inductance machinery rests on, and what the far field must still see.
    """
    gl, wl = np.polynomial.legendre.leggauss(quad)
    gt, wtq = np.polynomial.legendre.leggauss(quad_t)
    i, j, k = (a.reshape(-1) for a in np.meshgrid(np.arange(quad), np.arange(quad_t), np.arange(quad_t), indexing="ij"))
    n = len(ln)
    off = np.zeros((n, i.size, 3))
    other = np.stack([[c for c in range(3) if c != ax] for ax in axis])  # the two transverse axes
    rows = np.arange(n)[:, None]
    off[rows, np.arange(i.size)[None, :], axis[:, None]] = 0.5 * ln[:, None] * gl[i][None, :]
    off[rows, np.arange(i.size)[None, :], other[:, 0][:, None]] = 0.5 * wt[:, 0][:, None] * gt[j][None, :]
    off[rows, np.arange(i.size)[None, :], other[:, 1][:, None]] = 0.5 * wt[:, 1][:, None] * gt[k][None, :]
    w = ln[:, None] * (wl[i] * wtq[j] * wtq[k])[None, :] / 8.0
    return off, w


def bar_filaments(
    shape, size=None, quad: int = 3, quad_t: int = 2, sigma=None, freq: float = 0.0, grid_shapes=(), edges=None
):
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
        sigma: conductivity, one entry per conductor. Needed whenever conductors touch: a bar
            straddling two of them has half its length in each, so its conductivity is their series
            (harmonic) mean.

            An entry is a scalar, a **callable of position**, or a vector carrying one value per
            CELL of that conductor -- see :func:`resolve_sigma`. The last two make the material a
            design variable at cell resolution, which is what a density (SIMP) topology optimisation
            is. The lattice does not move with it: a cell whose conductivity goes to zero is still a
            cell, still joined by bars, and still counted as metal by the thickness runs behind the
            skin term. That is the ordinary fixed-mesh treatment and it is why a converged density
            has to be read back out as a shape rather than assumed to be one.

    Returns:
        :class:`Filaments`, with ``lattice`` describing the grid the FFT path needs.

    The cell count per axis is ``max(1, round(extent / size))``, so the pitch is the extent divided
    by a whole number rather than ``size`` exactly -- a lattice has to close on the box.
    """
    shapes = list(shape) if isinstance(shape, (list, tuple)) else [shape]
    # `grid_shapes` size the GRID without being meshed into it. Two physics on one geometry -- the
    # conductors and the core -- must share a lattice or the operator coupling them is not Toeplitz
    # and the FFT does not apply. Built independently they do not: separate calls take their extent
    # and their pitch from their own regions, so the grids come out different sizes AND offset from
    # each other. Passing each the other's shapes here makes both agree, while occupancy below still
    # sees only the shapes this mesh owns.
    frame = list(grid_shapes)
    declared = [sh._size for sh in shapes + frame if sh._size is not None]
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
    # `edges=` STATES the grid, so it needs no pitch to derive one from -- the requirement applies to
    # the uniform path alone.
    if edges is None and np.any(h <= 0):
        raise ValueError(
            "peec.bar_filaments: no cell pitch. Pass size=, or give the Shape a size= when you build "
            "it — a lattice count cannot be guessed from the geometry alone."
        )
    # Taking the minimum DISCARDS every coarser pitch that was asked for, and the whole layout is
    # then discretised at the finest one -- which is the most expensive option, arrived at by asking
    # for something cheaper. Silent, it reads as local refinement having worked. It has not: the FFT
    # needs one translation-invariant grid, so a solid lattice has exactly one pitch.
    if size is None and len(declared) > 1:
        want = [np.broadcast_to(np.asarray(v, float).reshape(-1), (3,)) for v in declared]
        if not all(np.allclose(w, want[0]) for w in want):
            names = [sh._region_name or "<unnamed>" for sh in shapes if sh._size is not None]
            asked = ", ".join(f"{n}={'x'.join(f'{1e3 * c:g}' for c in w)} mm" for n, w in zip(names, want))
            _warn_once(
                "peec.bar_filaments: these conductors ask for different cell pitches (%s), but solids "
                "share ONE lattice and therefore one pitch -- the FFT that applies the partial "
                "inductance needs a translation-invariant grid. The FINEST was used everywhere "
                "(%s mm), so the coarser requests cost more rather than less. To refine only part of "
                "a layout, model that part as a `Shape.line` -- a wire keeps its own discretisation "
                "and welds where the metal touches." % (asked, "x".join(f"{1e3 * c:g}" for c in h)),
            )
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for sh in shapes + frame:
        bnd = np.asarray(sh.bounds(), dtype=float).reshape(2, -1)
        b0, b1 = np.zeros(3), np.zeros(3)
        b0[: bnd.shape[1]], b1[: bnd.shape[1]] = bnd[0], bnd[1]
        lo, hi = np.minimum(lo, b0), np.maximum(hi, b1)
    ext = hi - lo
    ext = np.where(ext > 0, ext, np.where(h > 0, h, 1.0))  # a flat axis is one cell thick
    if isinstance(edges, GradedGrid):
        # resolved HERE rather than by the caller, because the extent, the base pitch and the faces
        # are all already in scope -- a caller would have to rederive all three to agree with them
        edges = _graded_edges(shapes + frame, lo, lo + ext, h, edges.fine, edges.halo)
    if edges is None:
        n = np.maximum(1, np.round(ext / h).astype(int))
        d = ext / n  # cell pitch per axis, closing exactly on the box
        # a GRADED grid is the same arrays with a per-cell pitch, so the whole of the rest of this
        # function is written against `dax` and the uniform case simply repeats one value
        dax = [np.full(int(n[a]), float(d[a])) for a in range(3)]
        org = lo.copy()
    else:
        # A GRADED RECTILINEAR grid: per-axis cell boundaries instead of one pitch.
        #
        # Cells stay axis-aligned boxes, so `bar_self` and `_volume_rule` are unchanged, and the grid
        # stays logically structured (i, j, k) -- so there are no hanging nodes, the incidence is the
        # same, and nothing needs a tie. What it gives up is translation invariance: the kernel no
        # longer depends on the index offset alone, so the FFT does NOT apply and the operator has to
        # be `jno.solve.hierarchical(...)`. What it buys is resolution where the features are, which
        # a single pitch cannot give -- every real layout here has a 1.0 mm minimum trace on a
        # 96 x 74 mm plate, and resolving it uniformly costs 1-3 M elements.
        if len(edges) != 3:
            raise ValueError(f"peec.bar_filaments: edges= needs one array of cell boundaries per axis, got {len(edges)}.")
        eax = [np.asarray(e, dtype=float).reshape(-1) for e in edges]
        for a, e in enumerate(eax):
            if e.size < 2 or not np.all(np.diff(e) > 0):
                raise ValueError(
                    f"peec.bar_filaments: edges[{a}] must be at least two STRICTLY INCREASING cell "
                    f"boundaries; got {e.size} values. A repeated boundary is a zero-width cell, which "
                    "has no volume and no self term."
                )
            # ...and a boundary that is merely ALMOST repeated is just as bad, which `> 0` does not
            # catch. Building edges by unioning two coordinate sets is the natural way to grade a
            # grid, and `np.unique` does not merge floats that differ in the last bit -- so 0.8 mm
            # from two different `linspace` calls yields a cell 1e-19 wide, zero volume, and a
            # singular operator reported from inside a preconditioner callback with nothing pointing
            # back here. Measured: exactly that, on the first graded solve attempted.
            step = np.diff(e)
            span = float(e[-1] - e[0])
            if span > 0 and step.min() < 1e-6 * span:
                k = int(np.argmin(step))
                raise ValueError(
                    f"peec.bar_filaments: edges[{a}] has a cell of width {step.min():.3e} between "
                    f"{e[k]:.9g} and {e[k + 1]:.9g}, which is {step.min() / span:.1e} of the axis span. "
                    "A cell that thin has no usable volume and makes the operator singular. If these "
                    "edges came from merging two coordinate sets, round them to a common tolerance "
                    "before taking the union -- `np.unique` keeps values that differ in the last bit."
                )
        dax = [np.diff(e) for e in eax]
        n = np.array([len(v) for v in dax], dtype=int)
        d = np.array([float(v.mean()) for v in dax])  # nominal, for reporting only
        org = np.array([float(e[0]) for e in eax])
        lo = org
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError(f"peec.bar_filaments: the box has a zero or non-finite extent {tuple(ext)}.")
    if int(n.max()) < 2:
        raise ValueError(
            f"peec.bar_filaments: a pitch of {h} puts one cell across every axis of this box, so no bar "
            "joins two cells and no current can flow. Use a smaller size=."
        )
    # cell centres per axis, from the cumulative boundaries -- identical to lo + (i + 0.5) * d when
    # the spacing is uniform, so the uniform path is unchanged to the last bit
    cax = [org[a] + np.cumsum(dax[a]) - 0.5 * dax[a] for a in range(3)]

    # A regular grid over the BOUNDING BOX, then a mask: the grid deliberately ignores the shape, so
    # an L-shaped trace or a slot is a pattern of absent cells rather than a distorted mesh. That is
    # what keeps the operator translation-invariant, and therefore what keeps the FFT applicable.
    ijk = np.stack(np.meshgrid(*[np.arange(v) for v in n], indexing="ij"), axis=-1).reshape(-1, 3)
    centres = np.stack([cax[a][ijk[:, a]] for a in range(3)], axis=1)
    cell_d = np.stack([dax[a][ijk[:, a]] for a in range(3)], axis=1)  # this cell's own size per axis
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
    # ...and the same question PER CONDUCTOR, which the check above does not ask. A lattice takes a
    # cell when its CENTRE is inside the geometry, so a solid thinner than the pitch can fall between
    # two rows of centres and contribute nothing at all -- while every other conductor meshes
    # normally and the network solves, with that piece simply absent.
    #
    # This is not hypothetical and it is not rare. On a power module whose 0.57 mm traces force a
    # one-cell-thick z pitch (which the surface-impedance guard REQUIRES at MHz), the 0.18 mm dies
    # land between cell centres at 3.34 / 3.86 / 4.39 mm and all four vanish. The solve then reports
    # a loop inductance for a circuit that has no switches in it, and nothing anywhere says so --
    # the same conductivity applied to the die changed the answer by not one bit, which is how it
    # was found.
    #
    # Raised rather than warned: a conductor that is in the drawing and not in the model is a
    # different circuit, not a less accurate one.
    lost = [si for si in range(len(shapes)) if not (own == si).any()]
    if lost:
        nm = [getattr(shapes[si], "_region_name", None) or f"#{si}" for si in lost]
        ext = np.asarray(shapes[lost[0]].bounds(), dtype=float).reshape(2, -1)
        thin = (ext[1] - ext[0]) / np.asarray(h)[: ext.shape[1]]
        raise ValueError(
            f"peec.bar_filaments: {len(lost)} conductor(s) {nm} got no cell of this lattice, so they "
            "would be absent from the network while everything else solved normally. A cell belongs "
            "to a solid when its CENTRE is inside it, so a solid thinner than the pitch can fall "
            f"between two rows of centres: {nm[0]!r} spans "
            f"{tuple(round(float(v), 6) for v in ext[1] - ext[0])} m against a pitch of {tuple(h)}, "
            f"which is {tuple(round(float(v), 2) for v in thin)} cells across.\n"
            "Either use a finer size= so the thin part gets a row of centres, or -- if a thin solid "
            "sits ON another and was meant to be part of it -- merge them, because a lattice cannot "
            "give one part its own pitch. Overlapping solids are the other cause: cells go to the "
            "FIRST solid containing them, so a piece drawn inside another never gets any."
        )
    nid = np.full(int(np.prod(n)), -1)
    nid[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
    nid = nid.reshape(tuple(n))
    nodes = centres[keep]
    node_d = cell_d[keep]  # each live cell's own size per axis; one value repeated when uniform
    cell_part = own[keep]

    # How thick the CONDUCTOR is across each direction, and into how many cells that is divided.
    # The thickness is the smaller transverse EXTENT, not the smaller pitch: a 0.57 mm trace on a
    # 0.5 mm in-plane grid is thin in z and wide in y, and picking by pitch would call the 0.5 mm
    # width the thickness and hand the skin formula the wrong dimension.
    # A cell's MATERIAL label: shapes declared with the same conductivity are one conductor, so two
    # stacked copper pieces stay a single run while a die on a trace does not. Falls back to the
    # shape index when the conductivities are not concrete -- a traced sigma in an inverse problem,
    # or a field, which varies WITHIN a conductor and so cannot label one.
    mat = np.asarray(own)
    if sigma is not None:
        try:
            sv = np.asarray([float(np.asarray(v)) for v in (sigma if isinstance(sigma, (list, tuple)) else [sigma])])
            if sv.size == len(shapes):
                mat = np.unique(sv, return_inverse=True)[1][np.maximum(own, 0)]
        except (TypeError, ValueError):
            pass
    lab = np.where(own < 0, -1, mat).reshape(tuple(n))
    runs = [_occupied_runs(lab, t) for t in range(3)]

    cen, tan, ln, area, ends, axis, owner, masks = [], [], [], [], [], [], [], {}
    skin, span = [], []
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
        t0, t1 = [i for i in range(3) if i != ax]
        # A bar joins two cells adjacent along `ax`, so they share their transverse extent and the
        # bar's cross-section is that cell's own. Its LENGTH is centre to centre, which is half of
        # each cell's size along the axis -- equal to the pitch when uniform, and not when graded.
        w, t = node_d[na][:, t0], node_d[na][:, t1]
        blen = 0.5 * (node_d[na][:, ax] + node_d[nb][:, ax])
        r0, r1 = runs[t0][a][live].reshape(-1), runs[t1][a][live].reshape(-1)
        # The conductor's thickness across each transverse direction. On a graded grid the run is a
        # COUNT of cells that need not be equal, so this takes the local cell size as representative
        # -- exact when uniform, and an approximation for the skin-depth gate otherwise. It decides
        # which impedance model applies, never a solved value.
        e0, e1 = r0 * w, r1 * t
        thin = e0 <= e1
        skin.append(np.where(thin, e0, e1))  # the conductor's THICKNESS, not the cell pitch
        span.append(np.where(thin, r0, r1))  # 1 means this element is the whole thickness
        cen.append(0.5 * (pa + pb))
        tan.append(u)
        ln.append(blen)
        area.append(w * t)
        ends.append((na, nb))
        axis.append(np.full(k, ax))
        owner.append(np.stack([cell_part[na], cell_part[nb]], axis=1))
    if not cen:
        raise ValueError(
            "peec.bar_filaments: no bar joins two cells of this conductor, so no current can flow. Use a smaller size=."
        )

    cen = np.concatenate(cen)
    skin = np.concatenate(skin)
    span = np.concatenate(span)
    tan = np.concatenate(tan)
    ln = np.concatenate(ln)
    area = np.concatenate(area)
    axis = np.concatenate(axis)
    owner = np.concatenate(owner)
    part = owner[:, 0]
    end_a = np.concatenate([ea for ea, _ in ends])
    end_b = np.concatenate([eb for _, eb in ends])
    nb = len(ln)

    # A conductor thick against the skin depth carries a current SHEET PER FACE rather than one
    # current spread through it -- otherwise its inductance contradicts its own surface impedance,
    # and a return plane's thickness moves L where it physically cannot. The pair is a second lattice
    # family on the same grid and masks, its current displaced to the opposite face; the split
    # between them is found by the solve, through `slab_transfer_impedance`.
    delta = _skin_depth(sigma, freq)
    sheets = _sheet_families(axis, skin, span, d, delta) if delta else {}
    pair = -np.ones(nb, dtype=int)
    sheet_thin = -np.ones(nb, dtype=int)
    sheet_ext = np.zeros(nb)
    if sheets:
        blocks, start = [], 0
        for ax in sorted(set(axis.tolist())):
            k = int((axis == ax).sum())
            idx = np.arange(start, start + k)
            if ax in sheets:
                thin, thick = sheets[ax]
                # the layer the current actually occupies: the whole half below 2 skin depths, and
                # the skin layer itself above it, so the sheet sits where the current does
                ext = float(min(0.5 * thick, 2.0 * delta))
                shift = 0.5 * (thick - ext)
                blocks.append((idx, +shift, ext, thin))
                blocks.append((idx, -shift, ext, thin))
            else:
                blocks.append((idx, 0.0, 0.0, -1))
            start += k
        take = np.concatenate([b[0] for b in blocks])
        offs = np.concatenate([np.full(len(b[0]), b[1]) for b in blocks])
        sheet_ext = np.concatenate([np.full(len(b[0]), b[2]) for b in blocks])
        sheet_thin = np.concatenate([np.full(len(b[0]), b[3], dtype=int) for b in blocks])
        pair, at = -np.ones(len(take), dtype=int), 0
        i = 0
        while i < len(blocks):
            k = len(blocks[i][0])
            if blocks[i][3] >= 0:  # a paired axis contributes its two sheets back to back
                pair[at : at + k] = np.arange(at + k, at + 2 * k)
                pair[at + k : at + 2 * k] = np.arange(at, at + k)
                at += 2 * k
                i += 2
            else:
                at += k
                i += 1
        cen, skin, span = cen[take], skin[take], span[take]
        tan, ln, area, axis = tan[take], ln[take], area[take], axis[take]
        owner, part = owner[take], part[take]
        end_a, end_b = end_a[take], end_b[take]
        nb = len(take)
        live = sheet_thin >= 0
        cen = cen.copy()
        cen[np.flatnonzero(live), sheet_thin[live]] += offs[live]
    # A bar joins two CELLS, and its two halves are in SERIES, so its conductivity is their harmonic
    # mean. That rule was written for a bar straddling two conductors -- a shorting strap touching
    # two plates is the normal case, not an error -- and it is the same rule a per-cell conductivity
    # needs, so both resolve here: the conductivity is fixed to the CELLS, and the bars read it.
    # For one conductivity per conductor this degenerates to exactly the old per-conductor form.
    # Resolved per conductor over ITS OWN cells, then scattered back: a field is evaluated where its
    # conductor actually is, and a per-cell vector is that conductor's cell count, not the whole
    # grid's -- the grid spans every conductor sharing it, which is not a design variable.
    #
    # Kept as a CLOSURE over the concrete pieces rather than a value, so the conductivity can be
    # re-resolved later without redoing any of the host work above. That is what lets `.build()`
    # freeze the discretisation once and still take a traced conductivity on every solve.
    _sel = [np.flatnonzero(cell_part == si) for si in range(len(shapes))]
    _inv = np.empty(len(nodes), dtype=int)
    _inv[np.concatenate(_sel) if _sel else np.zeros(0, dtype=int)] = np.arange(len(nodes))
    _names = [getattr(sh, "_region_name", None) or f"#{i}" for i, sh in enumerate(shapes)]

    def resolve(values):
        """One conductivity per conductor -> one per bar. Safe to call under a trace."""
        vals = list(values) if isinstance(values, (list, tuple)) else [values]
        if len(vals) != len(shapes):
            raise ValueError(f"peec.bar_filaments: {len(vals)} conductivities for {len(shapes)} conductors.")
        cell_sigma = jnp.concatenate(
            [jnp.asarray(resolve_sigma(v, nodes[sl], f"conductor {nm!r}")) for v, sl, nm in zip(vals, _sel, _names)]
        )[_inv]
        s0, s1 = cell_sigma[end_a], cell_sigma[end_b]
        if cell_sigma.ndim == 2:
            # An anisotropic material: every bar belongs to exactly one axis family, so it takes the
            # component of that axis from each of its two end cells. This is the whole of the
            # anisotropic path -- the lattice already knew which direction each bar runs in.
            ax = jnp.asarray(axis)[:, None]
            s0 = jnp.take_along_axis(s0, ax, axis=1)[:, 0]
            s1 = jnp.take_along_axis(s1, ax, axis=1)[:, 0]
        return 2.0 * s0 * s1 / (s0 + s1)

    bar_sigma = resolve(sigma) if sigma is not None else None
    if sigma is None and not np.all(owner[:, 0] == owner[:, 1]):
        bad = int(np.flatnonzero(owner[:, 0] != owner[:, 1])[0])
        raise ValueError(
            f"peec.bar_filaments: a bar joins cells of conductors {owner[bad, 0]} and {owner[bad, 1]}, so "
            "its conductivity depends on both. Pass sigma= (one per conductor) and it is resolved as the "
            "series combination."
        )

    # the two transverse sizes of each bar's own cell, which on a graded grid vary bar to bar
    wt = np.stack([np.array([node_d[e][i] for i in range(3) if i != int(ax)]) for e, ax in zip(end_a, axis)])
    if sheets:  # a sheet is as thick as the layer its current occupies, not as the conductor
        for r in np.flatnonzero(sheet_thin >= 0):
            cols = [c for c in range(3) if c != int(axis[r])]
            wt[r, cols.index(int(sheet_thin[r]))] = sheet_ext[r]
    # A lattice element is a CUBE, so the quadrature samples it like one. Points only along the axis
    # leave the cross-section a point, and neighbouring cells sit one pitch apart while extending a
    # full pitch transversely -- so every near-neighbour mutual was over-counted. Measured against
    # the volume integral: +1.2 % when the element is 8x longer than it is thick, +15.3 % at a cube,
    # +48.8 % at 8x shorter. `quad_t` is separate from `quad` because the two buy different things
    # and cost differently: the axis rule sets the along-length accuracy of FAR pairs, the transverse
    # rule fixes NEAR over-counting, and the sub-point count -- which the dense and welded paths pay
    # quadratically -- goes as quad * quad_t^2.
    sub, wsub = _volume_rule(ln, wt, axis, int(quad), int(quad_t))
    nsub = sub.shape[1]
    pos = (cen[:, None, :] + sub).reshape(-1, 3)
    mom = (tan[:, None, :] * wsub[:, :, None]).reshape(-1, 3)
    group = np.repeat(np.arange(nb), nsub)
    self_g = bar_self(ln, wt[:, 0], wt[:, 1])  # numpy in, numpy out: a host constant of the grid

    ir, ic, iv, off = [], [], [], 0
    for na, nb_ in ends:
        k = len(na)
        cols = np.arange(off, off + k)
        ir += [na, nb_]
        ic += [cols, cols]
        iv += [np.ones(k), -np.ones(k)]
        off += k
    inc = sp.coo_matrix((np.concatenate(iv), (np.concatenate(ir), np.concatenate(ic))), shape=(len(nodes), nb)).tocsr()

    return Filaments(
        jnp.asarray(pos),
        jnp.asarray(mom),
        jnp.asarray(self_g),
        group,
        inc,
        jnp.asarray(ln),
        jnp.asarray(area),
        jnp.asarray(nodes),
        part,
        jnp.asarray(skin),
        np.zeros(nb, dtype=bool),
        span,
        {
            "n": tuple(int(v) for v in n),
            "d": tuple(float(v) for v in d),
            # A GRADED grid is not translation invariant: the kernel depends on where the cells are,
            # not just on their index offset, so it is NOT block-Toeplitz and the FFT does not apply.
            # Carried explicitly rather than inferred, because every consumer of this dict assumes
            # uniformity and would otherwise be silently wrong rather than absent.
            "graded": edges is not None,
            "dax": None if edges is None else tuple(np.asarray(v) for v in dax),
            # The grid's ORIGIN -- the lower corner of cell (0,0,0)'s box. Two meshes are the same
            # lattice only if all three agree: same shape, same pitch, SAME ORIGIN. Shape and pitch
            # alone are not enough and it is not a corner case -- a conductor and a core of equal
            # extent sitting one above the other produce identical `n` and `d` on grids offset by
            # their whole separation, and a coupling convolved across them is silently wrong.
            "lo": tuple(float(v) for v in lo),
            "axis": axis,
            "masks": masks,
            # which CELLS carry material, as opposed to which BARS exist between them. The magnetic
            # potential matrix is a cell-to-cell operator, so it needs the occupancy the bar masks
            # are derived from rather than the bar masks themselves.
            "cells": (nid >= 0),
            "sigma": bar_sigma,
            # The conductivity RESOLVER, not just the value it produced: `.build()` keeps it and
            # calls it again per solve, so a design variable never re-runs the host discretisation.
            "resolve": resolve,
            # The FFT generator needs one length and one self term PER AXIS FAMILY, and on a lattice
            # both are constants of the grid. They used to be sampled out of the per-element arrays,
            # which quietly required those to be concrete -- so splicing a traced block into a welded
            # network (a moving bond wire on a fixed trace layer) failed inside the lattice half.
            # one self term per family is a constant of a UNIFORM grid; on a graded one every bar
            # has its own, so the shortcut is absent rather than wrong
            "self": (
                {}
                if edges is not None
                else {int(ax): float(np.asarray(self_g)[axis == ax][0]) for ax in sorted(set(axis.tolist()))}
            ),
            # {axis: (thin axis, conductor thickness, the layer each sheet's current occupies)}.
            # `lattice_apply` rebuilds the sub-points from this, so the FFT path and the assembled
            # one place the sheets identically -- they must, or they stop being the same operator.
            "sheets": {int(a): (int(v[0]), float(v[1]), float(min(0.5 * v[1], 2.0 * delta))) for a, v in sheets.items()},
        },
        pair,
    )


def lattice_apply(fil: Filaments, g, mu_scale: float = 1.0, quad: int = 3, quad_t: int = 2):
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
    if lat.get("graded"):
        raise ValueError(
            "peec.lattice_apply: this lattice is GRADED, so it is not translation invariant and the "
            "FFT does not apply to it -- the kernel depends on where the cells are, not only on their "
            "index offset. Apply it with `jno.solve.hierarchical(...)` instead, which needs no such "
            "structure. Refusing rather than returning the uniform-grid answer, which would be wrong "
            "everywhere the spacing changes and right everywhere else."
        )
    n, d, axis = lat["n"], lat["d"], np.asarray(lat["axis"])
    gx, gw = np.polynomial.legendre.leggauss(int(quad))

    masks = lat.get("masks") or {}
    sheets = lat.get("sheets") or {}
    fams, start = [], 0
    for ax in sorted(set(axis.tolist())):
        tot = int((axis == ax).sum())
        shape = tuple(v - 1 if i == ax else v for i, v in enumerate(n))
        length = float(d[ax])  # a lattice element IS the pitch; see the note on lattice["self"]
        cols = [c for c in range(3) if c != ax]
        wtx = np.array([d[c] for c in cols])
        sg = float(lat["self"][int(ax)])
        m = masks.get(ax)
        idx = None if m is None else jnp.asarray(np.flatnonzero(np.asarray(m).reshape(-1)))
        if int(ax) in sheets:
            # Two sheets on one grid: same masks, same self term, currents displaced to opposite
            # faces. AA and BB are the SAME operator -- a same-family generator sees sub[a] - sub[b],
            # so the constant face shift cancels -- which is why this costs three kernels, not four.
            thin, thick, ext = sheets[int(ax)]
            k = tot // 2
            wts = wtx.copy()
            wts[cols.index(int(thin))] = ext  # a sheet is as thick as the layer its current occupies
            base, wq = _volume_rule(np.array([length]), wts[None, :], np.array([ax]), int(quad), int(quad_t))
            base, w = base[0], wq[0]
            shift = np.zeros(3)
            shift[int(thin)] = 0.5 * (float(thick) - float(ext))
            sa, sb = base + shift, base - shift
            fams.append(
                (
                    (
                        lattice_operator(shape, d, g, sg, sub=base, w=w),
                        lattice_operator(shape, d, g, 0.0, sub=sa, w=w, sub_b=sb, w_b=w),
                        lattice_operator(shape, d, g, 0.0, sub=sa, w=w, sub_b=sb, w_b=w, transpose=True),
                    ),
                    slice(start, start + tot),
                    shape,
                    idx,
                    k,
                )
            )
            start += tot
        else:
            sub, wq = _volume_rule(np.array([length]), wtx[None, :], np.array([ax]), int(quad), int(quad_t))
            fams.append(
                (lattice_operator(shape, d, g, sg, sub=sub[0], w=wq[0]), slice(start, start + tot), shape, idx, None)
            )
            start += tot
    if start != len(axis):
        raise ValueError(f"peec.lattice_apply: the families cover {start} of {len(axis)} bars; this is a bug.")

    def _one(op, x, sh, idx):
        if idx is None:
            return op(x.reshape(sh)).reshape(-1)
        # Absent cells carry no current, so scatter into the FULL family grid, apply, and read
        # back only the live slots. Masking this way is what lets a hole or an L-shape keep the
        # translation invariance the FFT needs — the grid stays full, the current does not.
        # unique/sorted are true by construction (flatnonzero), and saying so is what makes the
        # scatter transposable — reverse mode refuses an unproven one.
        full = jnp.zeros(int(np.prod(sh)), x.dtype).at[idx].set(x, unique_indices=True, indices_are_sorted=True)
        return op(full.reshape(sh)).reshape(-1)[idx]

    def real_apply(cur):
        out = []
        for op, sl, sh, idx, k in fams:
            x = cur[sl]
            if k is None:
                out.append(_one(op, x, sh, idx))
            else:  # the two sheets of one slab: a symmetric 2x2 of Toeplitz blocks
                same, ab, ba = op
                xa, xb = x[:k], x[k:]
                out.append(
                    jnp.concatenate(
                        [_one(same, xa, sh, idx) + _one(ab, xb, sh, idx), _one(ba, xa, sh, idx) + _one(same, xb, sh, idx)]
                    )
                )
        return jnp.concatenate(out)

    def apply(cur):
        cur = jnp.asarray(cur)
        # Lp is real, and the circulant embedding uses rfftn, which will not take a complex array.
        # A complex current is therefore applied through its parts: Lp(a + ib) = Lp a + i Lp b.
        if jnp.iscomplexobj(cur):
            return mu_scale * (real_apply(jnp.real(cur)) + 1j * real_apply(jnp.imag(cur)))
        return mu_scale * real_apply(cur)

    return apply


def magnetic_potential_apply(fil: Filaments, mu0: float = 4e-7 * np.pi, quad: int = 2):
    """``apply(q) -> P_m q`` over the lattice's CELLS, by FFT -- the scalar dual of `lattice_apply`.

    The magnetic system reduces to ``(R_m + A' P_m A) I_fm = K_m I_c`` once the voxel potentials are
    eliminated, so `P_m` maps a magnetic charge on each cell to the scalar potential it produces.
    Its coefficient falls out exactly dual to the partial inductance:

        Lp  =  (mu0 / 4 pi) * <1/r> * (mom . mom)        vector, over BARS
        P_m =  (1 / 4 pi mu0) * <1/r>                    scalar, over CELLS

    (pypeec `lib_solver/system_matrix.py`: ``scale = 1 / (mu_0 dx^2 dy^2 dz^2)`` against a Green
    function ``vol^2 / (4 pi r)`` -- the volumes cancel and leave `1 / (4 pi mu0 r)`.)

    Two things are reused rather than rebuilt. The operator is the same block-Toeplitz FFT as the
    electric one, since cells sit on the same regular grid -- `lattice_kernel` never knew about
    moments, the moment contraction lives a level up. And the self term, the volume-averaged ``1/r``
    of a cell with itself, is precisely the six-fold integral `bar_self` already evaluates.
    """
    from .kernel import bar_self, lattice_operator

    lat = getattr(fil, "lattice", None)
    if lat is None or "cells" not in lat:
        raise ValueError(
            "peec.magnetic_potential_apply: these filaments carry no cell occupancy, so they did not "
            "come from a bar lattice. A magnetic material is discretised as a solid; a polyline has "
            "no cells for flux to divide between."
        )
    n, d = tuple(int(v) for v in lat["n"]), tuple(float(v) for v in lat["d"])
    live = jnp.asarray(np.flatnonzero(np.asarray(lat["cells"]).reshape(-1)))

    # sub-points over the cell VOLUME with weights summing to one: the double sum is then the
    # volume-averaged 1/r the coefficient above expects, with the cell volumes already cancelled.
    g1, w1 = np.polynomial.legendre.leggauss(int(quad))
    i, j, k = (a.reshape(-1) for a in np.meshgrid(*(np.arange(quad),) * 3, indexing="ij"))
    sub = np.stack([0.5 * d[0] * g1[i], 0.5 * d[1] * g1[j], 0.5 * d[2] * g1[k]], axis=1)
    w = (w1[i] * w1[j] * w1[k]) / 8.0
    self_g = float(bar_self(np.array([d[2]]), np.array([d[0]]), np.array([d[1]]))[0])
    op = lattice_operator(n, d, lambda r: 1.0 / r, self_g, sub=sub, w=w)
    scale = 1.0 / (4.0 * jnp.pi * mu0)

    def apply(q):
        q = jnp.asarray(q)
        full = jnp.zeros(int(np.prod(n)), q.dtype).at[live].set(q, unique_indices=True, indices_are_sorted=True)
        return scale * op(full.reshape(n)).reshape(-1)[live]

    return apply


def _bar_rule(d, axis, quad):
    """Sub-points over one lattice bar's volume, weights summing to ONE (an average, not a moment).

    Offset from the bar's LOWER cell centre, which is how a bar family is indexed: a bar along `axis`
    sits half a pitch further along it than the cell that names it.
    """
    g1, w1 = np.polynomial.legendre.leggauss(int(quad))
    i, j, k = (a.reshape(-1) for a in np.meshgrid(*(np.arange(quad),) * 3, indexing="ij"))
    off = np.stack([0.5 * d[0] * g1[i], 0.5 * d[1] * g1[j], 0.5 * d[2] * g1[k]], axis=1)
    off[:, axis] += 0.5 * d[axis]
    return off, (w1[i] * w1[j] * w1[k]) / 8.0


def coupling_generator(n, d, a, b, quad=2):
    """The BTTB generator coupling an electric bar family along `a` to a magnetic one along `b`.

    An electric element drives a magnetomotive force around a magnetic one by Ampere's law, so the
    kernel is the CURL of the Green function rather than the Green function itself:

        K[m, e] = (1 / 4 pi) L_e L_m < (e_a x r_hat) . e_b / r^2 >

    averaged over both element volumes. Equivalently `-d(1/r)/dx_c` with the Levi-Civita sign, which
    is the form pypeec builds by differencing two half-cell-shifted Green evaluations.

    Built here rather than through `lattice_kernel` because that takes a scalar function of DISTANCE
    and this needs the separation's components. `lattice_operator(generator=...)` then owns the
    embedding and the FFT, so only the kernel differs.

    Zero when ``a == b``: a bar drives no circulation around a face of its own orientation.
    """
    if a == b:
        return np.zeros(tuple(2 * int(v) for v in n))
    axes = []
    for ni in n:
        q = np.arange(2 * ni)
        axes.append(np.where(q < ni, q, q - 2 * ni))
    off = np.meshgrid(*axes, indexing="ij")
    valid = np.ones(off[0].shape, bool)
    for o, ni in zip(off, n):
        valid &= np.abs(o) < ni
    sep = [o * hi for o, hi in zip(off, d)]

    sub_a, w_a = _bar_rule(d, a, quad)
    sub_b, w_b = _bar_rule(d, b, quad)
    ea = np.eye(3)[a]
    eb = np.eye(3)[b]
    body = np.zeros(off[0].shape)
    for p in range(sub_a.shape[0]):
        for q in range(sub_b.shape[0]):
            ds = sub_b[q] - sub_a[p]  # from the electric element to the magnetic one
            r = [v + ds[i] for i, v in enumerate(sep)]
            rn = np.sqrt(sum(v * v for v in r))
            safe = np.where(rn > 0, rn, 1.0)
            rh = [v / safe for v in r]
            cross = np.cross(ea, np.stack(rh, axis=-1))  # (e_a x r_hat)
            body = body + w_a[p] * w_b[q] * np.where(rn > 0, (cross @ eb) / safe**2, 0.0)
    body = body * (d[a] * d[b]) / (4.0 * np.pi)
    return np.where(valid, body, 0.0)


def _grid_families(lat, what):
    """Where each axis family's elements sit in the FULL cell grid, as flat indices.

    A bar is named by its LOWER cell (see :func:`_bar_rule`), so a family along ``ax`` is a subset of
    the cell grid with the last slice along ``ax`` missing. Both the magnetic potential and the
    coupling are Toeplitz on the CELL grid rather than on a family's own shape, so each family has to
    be lifted into it -- a re-ravel of the same multi-index, and purely structural, so it stays on
    the host like every other "which element is where" question in this file.

    Returns ``[(axis, slice into the element vector, flat cell indices)]``, in the order
    :func:`bar_filaments` concatenates its families.
    """
    n = tuple(int(v) for v in lat["n"])
    axis = np.asarray(lat["axis"])
    if lat.get("sheets"):
        raise NotImplementedError(
            "peec: a magnetic system on a network discretised into current SHEETS is not supported. "
            "A sheet family carries its own sub-point offset, so the coupling to it is a different "
            "generator from the one built here -- and a coupling built on the wrong offsets is a "
            "plausible wrong answer, not a visible failure. The sheet model is off the front door."
        )
    masks = lat.get("masks") or {}
    out, start = [], 0
    for ax in sorted(set(axis.tolist())):
        tot = int((axis == ax).sum())
        shape = tuple(v - 1 if i == ax else v for i, v in enumerate(n))
        m = masks.get(ax)
        sub = np.arange(int(np.prod(shape))) if m is None else np.flatnonzero(np.asarray(m).reshape(-1))
        if sub.size != tot:
            raise ValueError(
                f"peec: the {what} family along axis {ax} holds {tot} elements for {sub.size} live "
                "lattice slots; this is a bug."
            )
        flat = np.ravel_multi_index(np.unravel_index(sub, shape), n)
        out.append((int(ax), slice(start, start + tot), jnp.asarray(flat)))
        start += tot
    if start != axis.size:
        raise ValueError(f"peec: the {what} families cover {start} of {axis.size} elements; this is a bug.")
    return out


def _same_lattice(fil: Filaments, mag: Filaments):
    """Both meshes on one grid, or the coupling between them is not Toeplitz and the FFT is a lie."""
    le, lm = getattr(fil, "lattice", None), getattr(mag, "lattice", None)
    if le is None or lm is None or "n" not in le or "n" not in lm:
        raise ValueError(
            "peec: a magnetic system needs both meshes on a bar lattice. A polyline's filaments are "
            "not Toeplitz, so there is no shared grid for the coupling to be a convolution on."
        )
    same = (
        tuple(le["n"]) == tuple(lm["n"])
        and np.allclose(le["d"], lm["d"])
        and np.allclose(le.get("lo", 0.0), lm.get("lo", 1.0))
    )
    if not same:
        raise ValueError(
            f"peec: the conductors sit on a {tuple(le['n'])} grid of pitch {tuple(le['d'])} at "
            f"{tuple(le.get('lo', ()))} and the core on a {tuple(lm['n'])} grid of pitch "
            f"{tuple(lm['d'])} at {tuple(lm.get('lo', ()))}. The coupling between them is Toeplitz "
            "only on a COMMON grid -- same shape, same pitch and same origin -- so build both with "
            "`bar_filaments(grid_shapes=...)`, each framed by the other's regions."
        )
    return tuple(int(v) for v in le["n"]), tuple(float(v) for v in le["d"])


def magnetic_coupling_apply(fil: Filaments, mag: Filaments, quad: int = 2):
    """``(mmf, flux)`` -- the two faces of ONE coupling block, applied by FFT.

    Ampere's law gives the magnetomotive force a conductor's current drives around a magnetic
    element, ``mmf = K I_c``; reciprocity gives the flux that magnetisation links back into the
    conductor, ``flux = K' I_m``. They are the same block applied the two ways round, which is what
    keeps the assembled system symmetric -- and why both come out of one generator here rather than
    from two independently-built matrices that could disagree.

    Structure: one Toeplitz block per (electric axis ``a``, magnetic axis ``b``), zero when
    ``a == b`` because ``e_a x r_hat`` has no component along ``e_a``. So six blocks, not nine, and
    :func:`~jno.utils.solver.kernel.lattice_operator` owns the embedding and the FFT for each --
    ``transpose=True`` on the same generator is the reciprocal direction, at no extra kernel.

    Both element vectors are lifted into the full CELL grid before the convolution, because a bar
    family's own shape is one cell short along its axis and the two families are short along
    DIFFERENT axes. See :func:`_grid_families`.
    """
    from .kernel import lattice_operator

    n, d = _same_lattice(fil, mag)
    ne, nm = int(np.asarray(fil.length).size), int(np.asarray(mag.length).size)
    fams_e = _grid_families(fil.lattice, "conductor")
    fams_m = _grid_families(mag.lattice, "core")
    ncell = int(np.prod(n))

    # One generator per orientation pair, built once. `a == b` is identically zero and is simply not
    # built -- a block of zeros costs a full FFT per apply and buys nothing.
    ops = {}
    for ax_e, _se, _ie in fams_e:
        for ax_m, _sm, _im in fams_m:
            if ax_e == ax_m or (ax_e, ax_m) in ops:
                continue
            gen = coupling_generator(n, d, ax_e, ax_m, quad)
            ops[(ax_e, ax_m)] = (
                lattice_operator(n, d, None, None, generator=gen),
                lattice_operator(n, d, None, None, generator=gen, transpose=True),
            )

    def _scatter(x, idx, dtype):
        return jnp.zeros(ncell, dtype).at[idx].set(x, unique_indices=True, indices_are_sorted=True)

    def mmf(cur):
        """The magnetomotive force each magnetic element sees, from every conductor current."""
        cur = jnp.asarray(cur)
        out = jnp.zeros(nm, cur.dtype)
        for ax_m, sl_m, idx_m in fams_m:
            acc = None
            for ax_e, sl_e, idx_e in fams_e:
                if ax_e == ax_m:
                    continue
                y = ops[(ax_e, ax_m)][0](_scatter(cur[sl_e], idx_e, cur.dtype).reshape(n)).reshape(-1)
                acc = y if acc is None else acc + y
            if acc is not None:
                out = out.at[sl_m].set(acc[idx_m])
        return out

    def flux(im):
        """The flux the magnetisation links into each conductor element -- the same block, transposed."""
        im = jnp.asarray(im)
        out = jnp.zeros(ne, im.dtype)
        for ax_e, sl_e, idx_e in fams_e:
            acc = None
            for ax_m, sl_m, idx_m in fams_m:
                if ax_e == ax_m:
                    continue
                y = ops[(ax_e, ax_m)][1](_scatter(im[sl_m], idx_m, im.dtype).reshape(n)).reshape(-1)
                acc = y if acc is None else acc + y
            if acc is not None:
                out = out.at[sl_e].set(acc[idx_e])
        return out

    return mmf, flux


def magnetic_system_apply(mag: Filaments, chi, mu0: float = 4e-7 * np.pi, quad: int = 2):
    """``(apply, diag)`` for the magnetic operator ``R_m + A' P_m A`` -- the dual of ``R + jw Lp``.

    The magnetic unknown is the magnetisation flux ``I_m = mu0 M A`` on each element. Two statements
    close the system, and both are the mirror of an electric one:

        R_m I_m - A' phi_m = mmf          the mmf balance along an element (Ohm's law's dual)
        phi_m = -P_m A I_m                the potential of the magnetic charge the divergence leaves

    The sign in the second is the one worth writing down. Magnetic charge is ``rho_m = -div M``, so
    a positive magnetisation flowing OUT of a cell leaves NEGATIVE charge there, while the incidence
    reports ``+I`` at the cell an element starts from. The two minus signs cancel on substitution and
    the operator comes out ``R_m + A' P_m A`` -- symmetric and positive, as an energy must be. Had
    the charge sign been dropped it would have come out ``R_m - A' P_m A``, which is a demagnetising
    field that ADDS to the magnetisation driving it: a core that amplifies itself.

    ``chi = mu_r - 1`` per element, already series-averaged along the element by the lattice.

    Args:
        mag: the magnetic mesh, from :func:`bar_filaments` over the ``mu_r`` regions.
        chi: susceptibility per element. Complex is allowed and is how core loss enters.
        quad: sub-points per cell axis in the potential's volume average.

    Returns ``(apply, diag)``. The diagonal is exact -- ``P_aa + P_bb - 2 P_ab`` with both ends
    cells of the same grid, so it is two constants of the pitch, not a sampled approximation.
    """
    from .kernel import bar_self, magnetic_reluctance

    lat = mag.lattice
    d = tuple(float(v) for v in lat["d"])
    axis = np.asarray(lat["axis"])
    # chi + 1 is mu_r, and `magnetic_reluctance` is the one place the constitutive law is written.
    # The round trip is deliberate: what the lattice series-averages along an element has to be chi
    # (reluctances add in series, so 1/chi does), and what the formula is stated in terms of is mu_r.
    Rm = magnetic_reluctance(mag.length, mag.area, jnp.asarray(chi) + 1.0, mu0)
    P = magnetic_potential_apply(mag, mu0, quad)
    inc = mag.incidence.tocoo()
    A = _bcoo(inc.data.astype(complex), inc.row, inc.col, inc.shape)
    AT = _bcoo(inc.data.astype(complex), inc.col, inc.row, inc.shape[::-1])

    def apply(im):
        im = jnp.asarray(im)
        return Rm.astype(im.dtype) * im + AT @ P(A @ im)

    # The diagonal, for the preconditioner. An element joins two cells of one grid, so its self terms
    # are the SAME constant and its mutual is the volume-averaged 1/r one pitch apart along its own
    # axis -- one number per axis, evaluated here rather than read back out of the operator.
    g1, w1 = np.polynomial.legendre.leggauss(int(quad))
    i, j, k = (a.reshape(-1) for a in np.meshgrid(*(np.arange(quad),) * 3, indexing="ij"))
    pts = np.stack([0.5 * d[0] * g1[i], 0.5 * d[1] * g1[j], 0.5 * d[2] * g1[k]], axis=1)
    wq = (w1[i] * w1[j] * w1[k]) / 8.0
    self_g = float(bar_self(np.array([d[2]]), np.array([d[0]]), np.array([d[1]]))[0])
    nb_g = {}
    for ax in sorted(set(axis.tolist())):
        shift = np.zeros(3)
        shift[int(ax)] = d[int(ax)]
        rr = np.linalg.norm((pts[:, None, :] + shift) - pts[None, :, :], axis=-1)
        nb_g[int(ax)] = float((wq[:, None] * wq[None, :] / rr).sum())
    scale = 1.0 / (4.0 * np.pi * mu0)
    lap = scale * np.array([2.0 * self_g - 2.0 * nb_g[int(a)] for a in axis])
    return apply, Rm + jnp.asarray(lap)


def _lattice_diag(fil: Filaments, mu0: float):
    """``Lp_aa`` for every bar -- the diagonal a Jacobi preconditioner needs, without forming ``Lp``.

    In jax rather than numpy so a TRACED geometry survives it: a moving bond wire welded to a fixed
    trace layer makes the moments tracers, and this sits on the path to the preconditioner. The
    values are only ever consumed by a host callback, which sees them concrete at run time.
    """
    grp = np.asarray(fil.group)
    mom = jnp.asarray(fil.mom)
    tot = jax.ops.segment_sum(mom, jnp.asarray(grp), num_segments=int(grp.max()) + 1)
    return jnp.asarray(fil.self_g) * (tot * tot).sum(1) * (mu0 / (4.0 * jnp.pi))


def _sub_slice(fil, lo, hi):
    """The sub-point rows of ``fil`` belonging to elements ``[lo, hi)``, and their local group labels.

    Both stay NUMPY. They are structural -- which sub-points belong to which element -- and routing
    them through jnp made them trace-bound, so the caller's `np.asarray` on the labels raised inside
    a jit. Indexing a jnp array with a numpy index array works exactly the same.
    """
    grp = np.asarray(fil.group)
    m = (grp >= lo) & (grp < hi)
    return np.flatnonzero(m), grp[m] - lo, int(hi - lo)


def cross_block(pos_a, mom_a, grp_a, na, pos_b, mom_b, grp_b, nb, g, scale=1.0, chunk=2048, budget=32_000_000):
    """The ELEMENT-level coupling ``Lp_ab``, built once — ``(na, nb)``, dense.

    Two conductors on different discretisations couple through a block that is neither Toeplitz nor
    square, so neither the FFT nor a lattice trick applies to it. It also does not need them, because
    it is THIN: one side is a bond wire of a few hundred filaments against a trace of a hundred
    thousand bars, so the block is small even when the lattice is not.

    Built once rather than evaluated per apply. Recomputing it inside the Krylov loop is what a
    matrix-free method is supposed to avoid, and here it was catastrophic: a 6,806-bar lattice solved
    in 0.213 s, and adding a SINGLE 19-filament wire took it to 33.4 s — 157x for 0.3 % more elements
    — because every application rebuilt a (57 x 20,418) kernel evaluation twice, inside a graph XLA
    then had to compile. As a stored block it is 1 MB and a matvec.

    The sub-point double sum is contracted into elements here, so the returned block is indexed by
    element and an apply is one dense matvec.

    BOTH sides are contracted inside the chunk loop. Contracting only ``b`` and concatenating first
    builds an ``(n_a_subpoints, nb)`` intermediate, which is larger than the result by the sub-point
    count -- twelve, once a lattice element samples its volume rather than its axis. That is a
    quadratic cost on a term that is supposed to be the thin one: on a 12,000-element module it
    allocated 2.7 GB to produce a 46 MB block, and the solve peaked at 14.7 GB against 18 GB of
    machine. Accumulating instead is exact, not an approximation -- a segment sum over a partition
    is additive across the chunks of that partition.
    """
    pa, ma = jnp.asarray(pos_a), jnp.asarray(mom_a)
    pb, mb = jnp.asarray(pos_b), jnp.asarray(mom_b)
    ga, gb = jnp.asarray(grp_a), jnp.asarray(grp_b)
    # The pair BLOCK is what has to be bounded, not either side on its own: `chunk` sets the `a`
    # side and the `b` side then takes whatever keeps `chunk_a * chunk_b` within `budget` pairs.
    # Chunking both at 2048 bounds the memory but makes the blocks far too small -- measured 27.7 s
    # against 4.5 s, all of it dispatch -- so the budget buys the memory back without the tiling.
    cb = max(1, int(budget // max(int(chunk), 1)))
    out = None
    for lo in range(0, pa.shape[0], chunk):
        hi = min(lo + chunk, pa.shape[0])
        acc = None
        # BOTH sides are chunked. Chunking only `a` bounds nothing when `b` is the big one, and in a
        # welded network it is exactly that way round: the thin side is the bond wire and the lattice
        # it welds to is the wide one. A (chunk, n_b_subpoints, 3) displacement then reached 6.96 GB
        # on a 12,000-element module -- with `r` and the kernel on top, a 14.6 GB peak against 18 GB
        # of machine, which is what put this solve on the OOM killer's list.
        for lo2 in range(0, pb.shape[0], cb):
            hi2 = min(lo2 + cb, pb.shape[0])
            # r^2 by expansion rather than a (chunk_a, chunk_b, 3) displacement: only the DISTANCE
            # is wanted, so the third axis is pure overhead -- three quarters of this term's memory
            # for nothing, and a matmul is quicker than a broadcast subtract besides. Both sides are
            # centred on the chunk first: the cancellation in |a|^2 + |b|^2 - 2a.b is set by the
            # coordinate MAGNITUDE, and centring makes that the chunk's own extent instead of the
            # model's distance from the origin.
            A, B = pa[lo:hi], pb[lo2:hi2]
            c = 0.5 * (A.mean(0) + B.mean(0))
            A, B = A - c, B - c
            r2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
            r = jnp.sqrt(jnp.clip(r2, 1e-300))
            sub = (ma[lo:hi] @ mb[lo2:hi2].T) * g(r)  # (chunk_a, chunk_b)
            part = jax.ops.segment_sum(sub.T, gb[lo2:hi2], num_segments=nb).T  # contract b
            acc = part if acc is None else acc + part
        part = jax.ops.segment_sum(acc, ga[lo:hi], num_segments=na)  # then a, once per a-chunk
        out = part if out is None else out + part
    return scale * out


def _hier_apply(fil: Filaments, g, mu_scale, operator):
    """``apply(cur)`` for a lattice with no translation invariance -- a GRADED grid.

    The whole operator through one hierarchical compression, with no FFT and no dense block. This is
    the path that makes local refinement affordable: a uniform grid fine enough for the narrowest
    trace is fine everywhere, and on a real layout that is 1-3 million elements.
    """
    from .hmatrix import build as hbuild
    from .hmatrix import materialize as hmat

    hm = hbuild(
        fil.pos,
        fil.mom,
        np.asarray(fil.group),
        g,
        tol=operator.tol,
        leaf=operator.leaf,
        eta=operator.eta,
    )
    return hmat(hm, fil.pos, fil.mom, fil.self_g, g, scale=mu_scale)


def welded_apply(fil: Filaments, g, mu_scale: float = 1.0, quad: int = 3, quad_t: int = 2, operator=None):
    """``apply(cur) -> Lp @ cur`` for a network of several discretisations, still without forming Lp.

    Each block keeps whatever structure it has -- a lattice by FFT, a set of filaments densely -- and
    the blocks couple through :func:`cross_apply`. So a trace layer of a hundred thousand bars stays
    O(N) while the bond wires landing on it stay exact.

    ``operator=jno.solve.hierarchical(...)`` compresses the two DENSE pieces -- a non-lattice block's
    own `pair_matrix` and every `cross_block` between parts. Those are what make welding expensive:
    a 6,806-bar lattice solves in 0.213 s and one welded 19-filament wire takes it to 33.4 s, and a
    12,000-element module allocated 2.7 GB to produce a 46 MB coupling block. The lattice blocks keep
    their FFT untouched -- it is exact and already O(N log N), so there is nothing to win there.
    """
    from .hmatrix import build as hbuild
    from .hmatrix import materialize as hmat
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
            gl,
            None,
            jnp.asarray(fil.length)[lo:hi],
            jnp.asarray(fil.area)[lo:hi],
            None,
            None,
            jnp.asarray(fil.skin)[lo:hi],
            np.asarray(fil.round_)[lo:hi],
            np.asarray(fil.span)[lo:hi],
            lat,
        )
        if lat is None:
            if operator is not None and operator.worth_it(cnt, cnt):
                # passed through as they are: `hmatrix.build` refuses a tracer by name, which is a
                # better error than whatever `np.asarray` would raise three frames deeper
                hm = hbuild(
                    sub.pos,
                    sub.mom,
                    np.asarray(sub.group),
                    g,
                    tol=operator.tol,
                    leaf=operator.leaf,
                    eta=operator.eta,
                )
                diag.append(hmat(hm, sub.pos, sub.mom, sub.self_g, g, scale=mu_scale))
            else:
                k = pair_matrix(sub.pos, sub.mom, g, sub.self_g, group=sub.group) * mu_scale
                diag.append(lambda x, k=k: k @ x)
        elif lat.get("graded"):
            # a graded block is not Toeplitz either, so it takes the same hierarchical path as an
            # unstructured one -- the FFT below would silently return the uniform-grid answer
            if operator is None:
                raise ValueError(
                    "peec.welded_apply: one of these parts is a GRADED lattice, which has no FFT. "
                    "Pass `operator=jno.solve.hierarchical(...)`."
                )
            diag.append(_hier_apply(sub, g, mu_scale, operator))
        else:
            diag.append(lattice_apply(sub, g, mu_scale=mu_scale, quad=quad, quad_t=quad_t))
        sel.append((lo, hi, rows, gl, cnt))

    cross = {}
    for i, (loi, hii, ri, gi, ci) in enumerate(sel):
        for j, (loj, hij, rj, gj, cj) in enumerate(sel):
            if j <= i:
                continue
            # jnp, NOT np: a welded wire whose SHAPE is a design variable arrives here as a tracer,
            # and `np.asarray` on it raises. The hierarchical path needs host arrays and says so
            # itself (`hmatrix.build` refuses a tracer by name); the dense path must stay traceable,
            # which is what `test_a_welded_wire_is_differentiable_in_its_shape` pins.
            pi, mi = jnp.asarray(fil.pos)[ri], jnp.asarray(fil.mom)[ri]
            pj, mj = jnp.asarray(fil.pos)[rj], jnp.asarray(fil.mom)[rj]
            if operator is not None and operator.worth_it(ci, cj):
                # ONE structure, applied both ways: a cross block appears in each part's row, and
                # building it twice could pick different pivots for the two directions -- the
                # operator would then stop being symmetric, which a partial inductance must be.
                hm = hbuild(pi, mi, gi, g, b=(pj, mj, gj), tol=operator.tol, leaf=operator.leaf, eta=operator.eta)
                cross[(i, j)] = (
                    hmat(hm, pi, mi, None, g, scale=mu_scale, b=(pj, mj)),
                    hmat(hm, pi, mi, None, g, scale=mu_scale, b=(pj, mj), transpose=True),
                )
            else:
                K = cross_block(
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
                cross[(i, j)] = (lambda x, K=K: K @ x, lambda x, K=K: K.T @ x)

    def apply(cur):
        cur = jnp.asarray(cur)
        out = [d(cur[lo:hi]) for d, (lo, hi, *_r) in zip(diag, sel)]
        for (i, j), (fwd, adj) in cross.items():
            out[i] = out[i] + fwd(cur[sel[j][0] : sel[j][1]])
            out[j] = out[j] + adj(cur[sel[i][0] : sel[i][1]])
        return jnp.concatenate(out)

    return apply


_THICKNESS_WARNED = set()  # (network, frequency, count) already reported; see the guard
_KRYLOV_CACHE = {}  # one compiled Krylov solve per (network, size, tolerance), reused across frequencies
_LU_HOLDER = {}  # the current host factorisation, so the callback closure does not change identity


def _holder_solve(r):
    """Apply the CURRENT Schur factorisation.

    Indirection through a holder keeps this function's identity fixed, which is what lets one compiled
    solve serve a whole frequency sweep. The holder is set immediately before the solve that reads it,
    so sequential solves are safe; two solves running CONCURRENTLY in one process would not be, and
    nothing here is reentrant.
    """
    return _LU_HOLDER["lu"].solve(np.asarray(r)).astype(r.dtype)


def _holder_solve_t(r):
    """Apply the TRANSPOSE of the current factorisation, for the tangent system.

    The adjoint solves ``M^T y = c``, and a preconditioner for ``M`` is not one for ``M^T`` -- it is
    ``P^T`` that is. Reusing ``P`` there is what made the tangent system look intractable. The plain
    transpose, not the conjugate one, because that is the convention jax's linear transpose uses.
    """
    return _LU_HOLDER["lu"].solve(np.asarray(r), trans="T").astype(r.dtype)


def _bcoo(val, row, col, shape):
    """A BCOO from COO triplets, for a matvec inside a traced body."""
    return jsparse.BCOO((jnp.asarray(val), jnp.stack([jnp.asarray(row), jnp.asarray(col)], axis=1)), shape=shape)


#: sub-point PAIRS held at once inside `near_block`; the displacement it builds is 24 bytes each.
_NEAR_PAIR_BUDGET = 4_000_000


def near_block(fil: Filaments, g, mu_scale=1.0, reach=2.0):
    """The NEAR-FIELD part of ``Lp`` as COO triplets: every pair closer than ``reach`` element sizes.

    NOT WIRED INTO THE SOLVE — see below. A preconditioner built on ``diag(Z)`` alone is weak, and
    weakest exactly where a thin conductor meets a thick one. Measured on a 440-bar lattice and a
    354-bar lattice welded to a bond wire, at a relative residual of 1e-8:

        diag(Z)          117 iterations (lattice)    393 (welded)
        near-field band   21                          27
        exact Z            1                           1

    So keeping about a percent of the entries recovers almost all of it, and the welded case stops
    being the hard one. The block is built from element CENTRES rather than from a formed ``Lp``, so
    it costs O(N) memory and never materialises the dense operator it approximates.

    What makes a welded network hard is NOT a property of its operator. Measured against a plain
    lattice and against two plates joined by a narrow neck on ONE lattice (same bottleneck topology,
    which converges just as fast as no bottleneck at all):

        eigenvalues, cond(P^-1 M), field of values, departure from normality, eigenvector
        conditioning -- all equal to within a few percent, several of them BETTER for the welded case

    and with a RANDOM right-hand side all three take a similar number of iterations (87, 76, 70).
    The difference is entirely in how the PORT excitation projects onto the operator's eigenmodes:

        directions carrying 90 % of b:   68 (lattice)   91 (neck)   142 (welded)
        iterations to 1e-8:              15             15           58

    Forcing every ampere through a six-filament wire excites a far richer set of modes than spreading
    it across a plate, and GMRES then needs a higher-degree polynomial to annihilate them. That is
    why no amount of improving the OPERATOR's conditioning helped, and why a band that compresses the
    whole spectrum does: it makes the residual polynomial cheap enough that the richness stops
    mattering.

    Why it is not used yet: putting the band in the (1,1) block while the SCHUR complement still
    comes from the diagonal makes the two inconsistent. That is survivable at a few hundred elements
    (welded: 584 applications -> 108) and fatal at a few thousand -- on 6,825 elements the solve made
    no progress at all, stalling at a relative residual of 3e-02, with the band's own factorisation
    verified exact to 2e-15. Making them consistent needs ``Z_near^-1 A'``, which is one solve per
    node AND turns the Schur complement dense: it trades the problem rather than solving it. This is
    kept because it is the measured foundation for whatever does.
    """

    # Read the geometry NON-differentiably. A preconditioner only has to accelerate, so it is built
    # from wherever the geometry currently is and no gradient runs through it -- which is what lets a
    # moving conductor (a bond wire being routed) reach this at all. Under `jax.grad` the primal is
    # concrete, so this is the current configuration and not a stale one.
    def cut(a):
        """One geometric array, on the host.

        Not routed through ``jnp`` first: inside a jit even a concrete array re-wrapped that way
        becomes trace-bound, and reading it back raises -- while the geometry of a BUILT network is
        already concrete and can simply be read. The ``stop_gradient`` path stays for the case it
        was written for, a traced geometry under ``jax.grad``, where the primal is still concrete.
        """
        if not isinstance(a, jax.core.Tracer):
            return np.asarray(a)
        try:
            return np.asarray(jax.lax.stop_gradient(a))
        except jax.errors.TracerArrayConversionError as e:
            raise ValueError(
                "peec: the near-field preconditioner is built on the host from concrete geometry, "
                "and a geometry that is traced INSIDE a jit has no value to read. Freeze it with "
                "`jno.peec(...).build()` and vary the conductivity instead, or drop the jit -- "
                "under jax.grad alone the primal geometry is still concrete."
            ) from e

    mom = cut(fil.mom)
    grp = np.asarray(fil.group)
    pos = cut(fil.pos)
    ne = int(grp.max()) + 1
    cen = np.zeros((ne, 3))
    np.add.at(cen, grp, pos)
    cen /= np.bincount(grp, minlength=ne)[:, None]
    tot = np.zeros((ne, mom.shape[1]))
    np.add.at(tot, grp, mom)
    size = float(np.mean(cut(fil.length)))
    rad = reach * size

    from scipy.spatial import cKDTree

    pairs = np.asarray(list(cKDTree(cen).query_pairs(rad)), dtype=int).reshape(-1, 2)
    if pairs.size == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    # The element-to-element term, summed over each pair's sub-points — vectorised over pairs, not
    # looped. Every element carries the same number of sub-points, so the whole thing is one einsum;
    # as a python loop over 154,000 pairs it cost 1.1 s at 6,800 elements and 2.1 s at 12,300, paid
    # again at every frequency of a sweep.
    order = np.argsort(grp, kind="stable")
    counts = np.bincount(grp, minlength=ne)
    if counts.min() != counts.max():
        raise ValueError(
            "peec.near_block: elements carry different numbers of sub-points, which the vectorised "
            "quadrature assumes. Discretise the whole network at one `quad`."
        )
    q = int(counts[0])
    P3 = pos[order].reshape(ne, q, 3)
    M3 = mom[order].reshape(ne, q, 3)
    # Chunked over PAIRS. The displacement is (n_pairs, q, q, 3), and q is twelve once a lattice
    # element samples its volume, so the whole-array form is n_pairs * 3456 bytes before numpy's
    # temporaries -- 3.4 GB on a 12,000-element module, for a result that is one scalar per pair.
    # The sum over a pair is independent of every other pair, so chunking is exact, not an
    # approximation. Same defect, same shape, as `cross_block` had.
    out = np.empty(len(pairs))
    step = max(1, int(_NEAR_PAIR_BUDGET // max(q * q, 1)))
    for lo in range(0, len(pairs), step):
        sl = slice(lo, min(lo + step, len(pairs)))
        pa, pb = P3[pairs[sl, 0]], P3[pairs[sl, 1]]
        ma, mb = M3[pairs[sl, 0]], M3[pairs[sl, 1]]
        d = pa[:, :, None, :] - pb[:, None, :, :]
        r = np.sqrt((d * d).sum(-1))
        out[sl] = mu_scale * (np.einsum("pik,pjk->pij", ma, mb) * g(r)).sum((1, 2))
    return pairs[:, 0], pairs[:, 1], out
