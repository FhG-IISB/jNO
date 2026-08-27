"""``jno.peec`` — partial-element circuits: the conductors, and what is impressed on them.

A partial-element method never meshes. A conductor becomes filaments carrying its centreline and
cross-section, and the operator is Ruehli's Neumann double integral over their pairs (Ruehli,
*Inductance calculations in a complex integrated circuit environment*, IBM J. Res. Dev. 16(5), 1972;
*Equivalent circuit models for three-dimensional multiconductor systems*, IEEE Trans. MTT 22(3),
1974). So the whole input is the geometry plus what is impressed on its terminals::

    i, v = e.peec_symbols()
    emag = jno.peec([
        v(*dcp) - v(*dcn) - 1.0,     # a volt across the DC link
        i(*ac) - 0.0,                # the load terminal is open
    ], freq=1e6)

Three constraint forms and only three -- a source ``v(A) - v(B) - g``, a fixed potential
``v(A) - g``, a fixed current ``i(A) - g`` -- which is the whole vocabulary of a port.

**Scope, up front.** Conductors are :meth:`jno.Shape.line` tubes today; a solid is not yet
discretised and says so. The solve is dense, so it is the small-network path. And each filament
carries ONE current, so skin effect WITHIN a conductor is not represented -- see ``freq``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .utils.solver.peec import (
    Filaments,
    bar_filaments,
    line_filaments,
    port_spec,
    solve_network,
    terminal_nodes,
)

__all__ = ["peec"]

MU0 = 4e-7 * np.pi


def _domain_of(constraints):
    for c in constraints:
        for var in (getattr(c, "_coord_vars", None) or {}).values():
            d = getattr(var, "_domain", None)
            if d is not None:
                return d
    raise ValueError(
        "jno.peec: no domain found on any constraint. Bind each terminal to its region first, as in "
        "`p = e.variable('DC+', split=True)[:3]`, then write the port on it."
    )


class PEECSolution:
    """What a solved network hands back.

    ``Z``/``R``/``L`` describe the source port; with an array ``freq`` each is an array over it.
    """

    def __init__(self, freq, cur, part, terms, port, res):
        self.freq = freq
        self.i = cur  # (n_filament,) or (n_freq, n_filament) complex
        self.partial = part  # (n_filament, n_filament) partial inductances, henry
        self._terms, self._port, self._R = terms, port, res

    @property
    def Z(self):
        """Terminal impedance of the source port, ohm."""
        return self._port

    @property
    def R(self):
        """Resistive part of the port impedance, ohm."""
        return jnp.real(self._port)

    @property
    def L(self):
        """Loop inductance from the field energy, ``I' Lp I / I_port^2``, henry.

        Computed from the energy rather than ``Im(Z)/w`` so it is defined at DC too, and so it stays
        the loop inductance the currents actually produce -- at a frequency where they redistribute,
        that is a different (smaller) number than the DC one, which is the effect worth seeing.
        """
        cur = jnp.atleast_2d(self.i)
        # I^H Lp I, not I^T Lp I: the currents are complex, and the magnetic energy is the HERMITIAN
        # form. The transpose form is complex, and its real part goes negative once the phases spread
        # -- an inductance a passive loop cannot have.
        num = jnp.real(jnp.einsum("fi,ij,fj->f", jnp.conj(cur), self.partial.astype(cur.dtype), cur))
        out = num / jnp.abs(jnp.atleast_1d(self._port_current)) ** 2
        return out if jnp.ndim(self._port) else out[0]

    @property
    def joule(self):
        """Ohmic dissipation at the solved excitation, ``sum R_k |I_k|^2``, watt."""
        cur = jnp.atleast_2d(self.i)
        out = jnp.einsum("k,fk->f", self._R, jnp.abs(cur) ** 2)
        return out if jnp.ndim(self._port) else out[0]

    def current(self, terminal):
        """Net current injected at ``terminal``, amp."""
        if terminal not in self._terms:
            raise ValueError(f"jno.peec: no terminal {terminal!r}. Known: {sorted(self._terms)}.")
        return self._terms[terminal]


class PEEC:
    """A partial-element problem: built from the constraint list, solved by :meth:`solve`."""

    def __init__(self, constraints, freq=0.0):
        self.constraints = list(constraints)
        if not self.constraints:
            raise ValueError("jno.peec: no constraints. A network with nothing impressed on it carries no current.")
        self.freq = np.atleast_1d(np.asarray(freq, dtype=float))
        self._scalar_freq = np.ndim(freq) == 0
        self.domain = _domain_of(self.constraints)
        self.sources, self.currents, self.grounds = port_spec(self.constraints)
        if len(self.sources) != 1:
            raise ValueError(
                f"jno.peec: {len(self.sources)} sources; the impedance readouts describe ONE port. Write "
                "exactly one `v(A) - v(B) - volts`, and express the rest as fixed potentials or currents."
            )

    def _discretise(self):
        regions = dict(getattr(self.domain, "_shape_regions", {}) or {})
        named = {t for s in self.sources for t in s[:2]} | {t for t, _ in self.currents} | {t for t, _ in self.grounds}
        preds = dict(getattr(self.domain, "_tag_predicates", {}) or {})
        # A terminal is a named SUBSET of a conductor, so `domain.tag` is its natural spelling: a tag
        # carries no material semantics and no declaration-order priority, so a pad may sit wholly
        # inside the conductor it marks. A region works too — the lookup reads its shape directly,
        # deliberately bypassing the first-declared-wins rule that applies to MATERIALS — but then the
        # pad must be declared before the conductor, or that rule subtracts it to nothing.
        terms = {}
        for t in sorted(named):
            if t in regions:
                terms[t] = regions[t]
            elif t in preds:
                terms[t] = _from_predicate(preds[t])
            else:
                raise ValueError(
                    f"jno.peec: {t!r} is neither a region nor a tag of this domain. Mark the terminal "
                    f"first — `d.tag({t!r}, <predicate>)` — then write the port on it. Known regions "
                    f"{sorted(regions)}, known tags {sorted(preds)}."
                )
        conductors = {n: sh for n, sh in regions.items() if n not in named}
        if not conductors:
            raise ValueError(
                "jno.peec: every region is named by a port, so there is no conductor left to carry current. "
                "A terminal marks part of a conductor; it is not a conductor of its own."
            )
        try:
            sig = self.domain.attached("sigma")
        except KeyError:  # nothing declared it at all, which is the same story as declaring it patchily
            sig = {}
        lines, line_sig, solids = [], [], []
        for n, sh in conductors.items():
            if n not in sig:
                raise ValueError(f"jno.peec: conductor {n!r} has no conductivity. Give it one: .attach(sigma=...).")
            node = getattr(sh, "_node", None)
            kind = type(node[1]).__name__ if (isinstance(node, tuple) and node and node[0] == "leaf") else None
            if kind == "Line":
                lines.append(sh)
                line_sig.append(float(sig[n]))
            elif kind == "Box":
                solids.append((sh, float(sig[n])))
            else:
                raise NotImplementedError(_shape_msg(n, kind))

        parts, owners = [], []  # (Filaments, per-filament sigma) and the shapes they came from
        if lines:
            fl = line_filaments(lines)
            parts.append((fl, np.asarray(line_sig)[np.asarray(fl.part)]))
            owners.append(lines)
        if solids:
            # ONE grid for every solid. Separate lattices couple through a block that is not Toeplitz,
            # so a shared grid is what keeps the whole trace layer a single FFT: on the example module
            # that is ten coplanar traces of equal thickness, which is the case it fits exactly.
            shs = [sh for sh, _ in solids]
            fb = bar_filaments(shs)
            parts.append((fb, np.asarray([sg for _, sg in solids])[np.asarray(fb.part)]))
            owners.append(shs)
        return (*_weld(parts, owners), terms)

    def solve(self):
        """Solve at every frequency and return a :class:`PEECSolution`."""
        fil, sigma, terms = self._discretise()  # sigma already spread onto the filaments by provenance
        nodes = {t: terminal_nodes(fil, sh) for t, sh in terms.items()}

        cur, port, drive, inject = [], [], [], []
        for f in self.freq:
            c, _phi, inj = solve_network(
                fil, sigma, nodes, self.sources, self.grounds, self.currents, omega=2 * np.pi * float(f)
            )
            cur.append(c)
            inject.append(inj)
            a, _b, g = self.sources[0]
            drive.append(inj[a])
            port.append(g / inj[a])
        from .utils.solver.kernel import pair_matrix

        part = pair_matrix(fil.pos, fil.mom, lambda r: 1.0 / r, fil.self_g, group=fil.group) * (MU0 / (4 * jnp.pi))
        res = jnp.asarray(fil.length) / (jnp.asarray(sigma) * jnp.asarray(fil.area))

        cur = jnp.stack(cur)
        port = jnp.stack([jnp.asarray(p) for p in port])
        sol = PEECSolution(
            self.freq[0] if self._scalar_freq else self.freq,
            cur[0] if self._scalar_freq else cur,
            part,
            {
                t: (
                    jnp.stack([jnp.asarray(m[t]) for m in inject])[0]
                    if self._scalar_freq
                    else jnp.stack([jnp.asarray(m[t]) for m in inject])
                )
                for t in terms
            },
            port[0] if self._scalar_freq else port,
            res,
        )
        drive = jnp.stack([jnp.asarray(x) for x in drive])
        sol._port_current = drive[0] if self._scalar_freq else drive
        return sol


class _from_predicate:
    """A ``domain.tag`` predicate, wearing the ``.contains(points)`` face :func:`terminal_nodes` reads."""

    def __init__(self, pred):
        self._pred = pred

    def contains(self, pts):
        pts = np.asarray(pts)
        return np.asarray(self._pred(*[pts[:, i] for i in range(pts.shape[1])])).reshape(-1)


def _shape_msg(name, kind):
    return (
        f"jno.peec: conductor {name!r} is a {kind or 'CSG'} plan, and only two discretisations exist: a "
        "Shape.line becomes filaments along its centreline, a Shape.box becomes a lattice of bars. Build "
        "the conductor from those, or ask for the shape you need."
    )


def _weld(parts, owners):
    """One network from several discretisations, JOINED where the conductors are in contact.

    Each conductor is discretised on its own, so its node numbering is local: welding renumbers and
    stacks the incidences block-diagonally. That alone leaves the parts electrically separate, and a
    bond wire landing on a trace would carry no current -- so nodes are then tied where the geometry
    says the metal touches: a node lying INSIDE another conductor joins that conductor's nearest node.

    Contact is read from the geometry, never invented. Two conductors that merely pass close by stay
    separate, which is what a clearance is for.
    """
    if len(parts) == 1:
        return parts[0]
    fils = [f for f, _ in parts]
    incs = [np.asarray(f.incidence) for f in fils]
    inc = np.zeros((sum(a.shape[0] for a in incs), sum(a.shape[1] for a in incs)))
    r = c = 0
    for a in incs:
        inc[r : r + a.shape[0], c : c + a.shape[1]] = a
        r, c = r + a.shape[0], c + a.shape[1]
    shift = lambda key: _renumber([np.asarray(getattr(f, key)) for f in fils])
    cat = lambda key: jnp.concatenate([jnp.asarray(getattr(f, key)) for f in fils])
    nodes = np.concatenate([np.asarray(f.nodes) for f in fils])
    bounds = np.cumsum([0] + [len(np.asarray(f.nodes)) for f in fils])
    inc, nodes = _join_contacts(inc, nodes, bounds, owners)
    welded = Filaments(
        cat("pos"),
        cat("mom"),
        cat("self_g"),
        shift("group"),
        jnp.asarray(inc),
        cat("length"),
        cat("area"),
        jnp.asarray(nodes),
        shift("part"),
        # Each block keeps whatever structure it had — a lattice stays a lattice — and the blocks
        # couple through a cross term. That is what lets a trace layer stay an FFT while the bond
        # wires landing on it stay exact.
        {"welded": _spans(fils)},
    )
    return welded, np.concatenate([np.asarray(g) for _f, g in parts])


def _join_contacts(inc, nodes, bounds, owners):
    """Tie nodes that lie inside another conductor to that conductor's nearest node."""
    root = np.arange(len(nodes))

    def find(a):
        while root[a] != a:
            root[a] = root[root[a]]
            a = root[a]
        return a

    for i in range(len(bounds) - 1):
        mine = np.arange(bounds[i], bounds[i + 1])
        for j in range(len(bounds) - 1):
            if i == j:
                continue
            theirs = np.arange(bounds[j], bounds[j + 1])
            inside = np.zeros(len(mine), dtype=bool)
            for sh in owners[j]:
                inside |= np.asarray(sh.contains(nodes[mine])).reshape(-1)
            for k in np.flatnonzero(inside):
                a = mine[k]
                near = theirs[int(np.argmin(((nodes[theirs] - nodes[a]) ** 2).sum(1)))]
                ra, rb = find(a), find(near)
                if ra != rb:
                    root[ra] = rb
    lab = np.array([find(a) for a in range(len(nodes))])
    keep, inverse = np.unique(lab, return_inverse=True)
    merged = np.zeros((len(keep), inc.shape[1]))
    np.add.at(merged, inverse, inc)
    return merged, nodes[keep]


def _spans(fils):
    """``(lo, hi, lattice)`` per welded block, in element numbering."""
    out, off = [], 0
    for f in fils:
        k = len(np.asarray(f.length))
        out.append((off, off + k, getattr(f, "lattice", None)))
        off += k
    return out


def _renumber(blocks):
    """Concatenate integer label blocks, offsetting each so labels stay unique across them."""
    out, off = [], 0
    for b in blocks:
        out.append(b + off)
        off += int(b.max()) + 1
    return np.concatenate(out)


def peec(constraints, *, freq=0.0):
    """Build a partial-element circuit problem from its port constraints.

    Args:
        constraints: the ports — ``v(A) - v(B) - g``, ``v(A) - g``, ``i(A) - g``.
        freq: frequency in Hz, scalar or array (``omega = 2 pi f`` inside). An array solves at each
            and every port readout becomes an array over it.

            **The filament self-term is the DC geometric mean distance**, so the skin effect WITHIN
            a filament is not represented. Redistribution BETWEEN conductors is (that is what the
            method is for); to capture it inside one, split a conductor that is wide against the
            skin depth into filaments across its section.

    Returns:
        :class:`PEEC` — call ``.solve()``.
    """
    return PEEC(constraints, freq=freq)
