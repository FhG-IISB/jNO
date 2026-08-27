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

from .utils.solver.peec import line_filaments, port_spec, solve_network, terminal_nodes

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
        missing = sorted(named - set(regions))
        if missing:
            raise ValueError(f"jno.peec: constraints name {missing}, which are not regions of this domain.")
        # A region named by a port is a TERMINAL; everything else is a conductor to discretise.
        conductors = {n: sh for n, sh in regions.items() if n not in named}
        if not conductors:
            raise ValueError(
                "jno.peec: every region is named by a port, so there is no conductor left to carry current. "
                "A terminal is a pad ON a conductor, declared as its own region."
            )
        try:
            sig = self.domain.attached("sigma")
        except KeyError:  # nothing declared it at all, which is the same story as declaring it patchily
            sig = {}
        shapes, sigmas = [], []
        for n, sh in conductors.items():
            node = getattr(sh, "_node", None)
            if not (isinstance(node, tuple) and node and node[0] == "leaf" and type(node[1]).__name__ == "Line"):
                raise NotImplementedError(_solid_msg(n))
            if n not in sig:
                raise ValueError(f"jno.peec: conductor {n!r} has no conductivity. Give it one: .attach(sigma=...).")
            shapes.append(sh)
            sigmas.append(float(sig[n]))
        fil = line_filaments(shapes)
        return fil, np.asarray(sigmas)[np.asarray(fil.part)], {n: regions[n] for n in named}

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
        res = jnp.asarray(fil.length) / (jnp.asarray(sigma) * jnp.pi * jnp.asarray(fil.radius) ** 2)

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


def _solid_msg(name):
    return (
        f"jno.peec: region {name!r} is not a Shape.line, and only line conductors are discretised into "
        "filaments today. A solid needs a bar lattice (the kernel has bar_self and an FFT operator for "
        "it, but nothing builds one yet). Model the conductor as a polyline tube, or wait for the "
        "lattice path."
    )


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
