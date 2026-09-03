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

Four constraint forms and only four -- a source ``v(A) - v(B) - g``, a fixed potential ``v(A) - g``,
a fixed current ``i(A) - g``, and a two-terminal device ``v(A) - v(B) - Z*i(A)``. The first three are
the whole vocabulary of a port; the fourth is Ohm's law, and it is what puts a COMPONENT in the
network without meshing it as metal::

    v(*m_d) - v(*m_s) - 5e-3 * i(*m_d),      # a MOSFET's on-resistance

A device parameter belongs in the circuit, not in the geometry: voxelise a 0.18 mm die on a 0.285 mm
grid and its resistance follows the grid, which is both wrong and the reason the grid had to be fine.

The material is the other half of the input, and it is a **design variable as often as it is a
constant**. ``.attach(sigma=...)`` takes three spellings, and a gradient flows back through all
three::

    .attach(sigma=5.8e7)                             # a material
    .attach(sigma=lambda x, y, z: SIG * rho(x, y))   # a FIELD: the density is the design
    .attach(sigma=SIG * rho)                         # one value per element

A callable is evaluated at each element -- cell centres for a solid's lattice, midpoints for a wire
-- and its arity is positional, exactly as an attached FEM coefficient's is, so ``lambda x, y`` is a
planar field. That is the usual one: a trace is thin, and its material varies across the board
rather than through the 0.57 mm of it. Reach for the callable over the vector, because it says
nothing about the pitch and so survives a change of ``size=``, which a per-element vector cannot.

A field is what makes a density (SIMP) topology optimisation expressible here. What it does NOT do
is move the geometry: the lattice is fixed, so a cell whose conductivity goes to zero is still a
cell, still joined by bars, and still counted as metal by the thickness runs behind the skin term.
That is the ordinary fixed-mesh treatment, and it is why a converged density has to be read back out
as a shape rather than assumed to be one.

A design loop wants the geometry decided ONCE. Everything structural -- which cells are metal,
which nodes a pad owns, which filaments weld to which -- is host work on concrete geometry, and
``solve()`` redoes all of it every call, which is why it differentiates but does not jit.
:meth:`peec.build` is that pass done once::

    emag = jno.peec([v(*a) - v(*b) - 1.0], freq=1e6).build()      # host, once
    loss = jax.jit(lambda rho: emag.solve(sigma={"trace": f(rho)}).L)

What comes back solves in pure jax, so it jits and composes into ``jno.core`` through ``jno.fn``
exactly as ``fem.solve()`` does -- which is what puts a PEEC objective behind ``.optimizer()``,
``jno.optimizers.mma`` and ``jno.le``. Measured on a 1,540-bar plate: **42x** on the value and 23x
on the gradient, because the host pass, not the solve, was the cost. The same split as
:meth:`jno.precond.ams.build` and :class:`jno.trace.FemLinearSystem`.

**Scope, up front.** A conductor is either a :meth:`jno.Shape.line` tube or a closed-form solid,
which voxelises onto a lattice shared with every other solid on it. A network containing a lattice
is applied matrix-free -- that block by FFT -- and solved by GMRES; a network of wires alone has no
such structure, forms the dense operator, and is therefore the small-network path. Each filament
carries ONE current and its self-term is the DC geometric mean distance, so the skin effect WITHIN a
filament is not represented -- see ``freq``. A built network freezes its GEOMETRY: a design variable
may change a conductivity, never a shape, so build again for a new one.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sparse

from .utils.solver.peec import (
    Filaments,
    bar_filaments,
    element_centres,
    line_filaments,
    port_spec,
    resolve_sigma,
    solve_network,
    terminal_nodes,
)

__all__ = ["peec"]

MU0 = 4e-7 * np.pi

#: Quadrature the discretisation is built at: `quad` points along an element and `quad_t` across it.
#: They must agree between the line and bar halves of a welded network — see `_discretise`.
_QUAD, _QUAD_T = 3, 2


def _unit_permeability(v) -> bool:
    """Whether a declared ``mu_r`` is concretely 1, so the region is air.

    Only a CONCRETE one: a traced permeability has no value to compare, and a design variable that
    happens to pass through 1 would divide by a zero susceptibility. That edge is documented rather
    than guarded, because guarding it would mean changing the mesh mid-optimisation.
    """
    try:
        return bool(np.asarray(v).shape == () and float(np.asarray(v)) == 1.0)
    except (TypeError, ValueError):
        return False


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


class Dissipation(dict):
    """``{region: W/m^3}`` of heat, and the same split by the property that CAUSED it.

    The mapping itself is the total, so the thermal side is unchanged and a region that dissipates
    two ways never contributes half of itself by accident::

        q = d.by_region(emag.dissipation(), default=0.0)

    ``.sigma`` is the ohmic loss and ``.mu_r`` the core loss, each named after what the region
    attached -- the same spelling in and out, as with ``.attach(k=...)`` and ``d.k``, so no channel
    needs a vocabulary invented for it. A region declaring both appears in both, under its own name.
    """

    def __init__(self, channels):
        self._channels = dict(channels)
        total: dict = {}
        for ch in self._channels.values():
            for region, q in ch.items():
                total[region] = total[region] + q if region in total else q
        super().__init__(total)

    def _channel(self, name):
        ch = self._channels.get(name) or {}
        if not ch:
            have = sorted(k for k, v in self._channels.items() if v)
            raise ValueError(
                f"jno.peec: nothing in this model dissipates through {name!r}, so there is no "
                f"{name} loss to read. Dissipating properties here: {have or 'none'}. Attach "
                f"{name}= on the region that should carry it."
            )
        return dict(ch)

    @property
    def sigma(self):
        """``{region: W/m^3}`` of OHMIC loss -- the conductors, from the currents solved for."""
        return self._channel("sigma")

    @property
    def mu_r(self):
        """``{region: W/m^3}`` of CORE loss -- the magnetisation working against a complex ``mu_r``.

        A real permeability is lossless and there is nothing here; the imaginary part of ``mu_r`` is
        the lossy component, exactly as a complex permittivity carries dielectric loss.
        """
        return self._channel("mu_r")


class PEECSolution:
    """What a solved network hands back.

    ``Z``/``R``/``L`` describe the source port; with an array ``freq`` each is an array over it.
    """

    def __init__(
        self,
        freq,
        cur,
        fil,
        terms,
        port,
        res,
        vol=None,
        owner=None,
        names=(),
        link=None,
        pot=None,
        pads=None,
        mag=(None, None, None, None, ()),
    ):
        self._vol, self._owner, self._names = vol, owner, tuple(names)
        #: nodal potentials, and the node set each terminal owns -- what `voltage` reads. Kept
        #: because an OPEN terminal carries no current, so `current`/`Z` say nothing about it: the
        #: induced voltage on a transformer's unloaded secondary is only in `phi`.
        self._pot, self._pads = pot, dict(pads or {})
        #: the magnetic mesh's own breakdown -- current, reluctance, volume, owner, names. It is a
        #: SECOND mesh, so it needs all five: a region that is both conductor and core is meshed
        #: twice and its two losses are summed per unit of each mesh's own volume.
        self._mag_cur, self._mag_res, self._mag_vol, self._mag_owner, self._mag_names = mag
        #: total flux linkage per element, magnetisation included -- set only when a core is in the
        #: model. `L` reads it instead of `Lp I`, which is the same number when there is no core.
        self._link = link
        self.freq = freq
        self.i = cur  # (n_filament,) or (n_freq, n_filament) complex
        self._fil = fil  # the discretisation; the partial-inductance MATRIX is formed only on demand
        self._terms, self._port, self._R = terms, port, res

    @property
    def partial(self):
        """The dense ``(n, n)`` partial inductances, henry — formed HERE, not at solve time.

        It is the one object in a partial-element method that is never sparse, and the solve is
        built specifically to avoid it: a lattice applies it by FFT and a welded network by blocks.
        Forming it anyway to compute one readout cost more than everything else put together --
        measured on the example module at a 2 mm pitch, 3.7 GB of a 5.3 GB peak, against 0.84 GB for
        the solve itself. It is quadratic in the SUB-POINT count, not the element count, so three
        Gauss points per element make it nine times worse again.

        Ask for it when you want the matrix. :attr:`L` no longer does.
        """
        from .utils.solver.kernel import pair_matrix

        f = self._fil
        return pair_matrix(f.pos, f.mom, lambda r: 1.0 / r, f.self_g, group=f.group) * (MU0 / (4 * jnp.pi))

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

        With a CORE the flux a circuit links is not ``Lp I`` alone -- the magnetisation links its own,
        and that is the entire reason for putting a core there. The energy is then ``Re(I^H Lambda)``
        against the total linkage, which the coupled solve already knows. It is the same number as
        ``I^H Lp I`` when there is no core, exactly and not approximately: the circuit equation says
        ``j w Lambda = A' phi - Z_int I``, which reduces to ``j w Lp I`` with the coupling absent.
        """
        from .utils.solver.kernel import pair_quadratic
        from .utils.solver.peec import lattice_apply, welded_apply

        f = self._fil
        cur = jnp.atleast_2d(self.i)
        scale = MU0 / (4 * jnp.pi)
        if self._link is not None:
            lam = jnp.atleast_2d(self._link)
            num = jnp.stack([jnp.vdot(c, lk).real for c, lk in zip(cur, lam)])
            out = num / jnp.abs(jnp.atleast_1d(self._port_current)) ** 2
            return out if jnp.ndim(self._port) else out[0]

        # I^H Lp I, not I^T Lp I: the currents are complex, and the magnetic energy is the HERMITIAN
        # form. The transpose form is complex, and its real part goes negative once the phases spread
        # -- an inductance a passive loop cannot have.
        #
        # Computed as a QUADRATIC FORM, never through the matrix. Lp is real and symmetric, so
        # I^H Lp I = Re(I)' Lp Re(I) + Im(I)' Lp Im(I) exactly -- the cross terms cancel by symmetry.
        #
        # Each half is `x . (Lp x)` through the SAME apply the matrix-free solve is built on. It used
        # to be `pair_quadratic`, which also never forms the matrix but does walk every pair, and so
        # is O(N^2) behind a solve that is linear in the bars. Measured, solve against readout:
        #
        #     bars      6,688    23,688    57,472
        #     solve      0.24 s    0.79 s    2.29 s
        #     pair sum   1.43 s   16.2  s   76.5  s
        #
        # -- the inductance cost thirty times the answer it was reporting on. A lattice's Lp is
        # block-Toeplitz, so `lattice_apply` is the same quadrature in O(N log N); `test_peec_fft`
        # already pins that apply against the dense `pair_matrix` on every case, which is what makes
        # this a change of EVALUATION and not of value.
        lat = getattr(f, "lattice", None)
        welded = isinstance(lat, dict) and "welded" in lat
        structured = lat is not None and (not welded or any(b[2] is not None for b in lat["welded"]))

        if structured:
            ap = (welded_apply if welded else lattice_apply)(f, lambda r: 1.0 / r, mu_scale=scale)

            def _energy(x):
                return jnp.vdot(x, ap(x)).real
        else:
            # A polyline's filaments are not Toeplitz and welding several conductors breaks the
            # structure even when each part had it, so there is nothing to exploit. The pair sum
            # stays: it is chunked, never holds an (N, N) array, and rematerialises on the reverse
            # pass, so the gradient is bounded too.
            grp = jnp.asarray(f.group)

            def _energy(x):
                m = jnp.asarray(f.mom) * x[grp][:, None]
                return pair_quadratic(f.pos, m, lambda r: 1.0 / r, f.self_g, group=grp) * scale

        num = jnp.stack([_energy(jnp.real(c)) + _energy(jnp.imag(c)) for c in cur])
        out = num / jnp.abs(jnp.atleast_1d(self._port_current)) ** 2
        return out if jnp.ndim(self._port) else out[0]

    @property
    def joule(self):
        """Ohmic dissipation at the solved excitation, ``sum R_k |I_k|^2``, watt."""
        cur = jnp.atleast_2d(self.i)
        out = jnp.einsum("k,fk->f", self._R, jnp.abs(cur) ** 2)
        return out if jnp.ndim(self._port) else out[0]

    def dissipation(self):
        """``{region: W/m^3}`` — the ohmic loss of each conductor, per unit of its own volume.

        Shaped for :meth:`jno.domain.by_region`, which is how a per-region quantity enters a weak
        form, so the thermal side reads::

            q = d.by_region(emag.dissipation(), default=0.0)
            heat = d.k * grad(T) . grad(s) - q * s

        Volumetric rather than total because that is what a source term is. The volume is the
        DISCRETISATION's -- the summed filament volumes -- so it is consistent with the currents that
        produced the loss rather than with the analytic solid, which a faceted mesh would not match.

        Which elements a conductor owns is STRUCTURAL and is read from ``_owner`` on the host; only
        the loss and the volume are jnp. That is what makes this callable inside a jit, and so what
        lets a thermal objective built on it be driven by ``jno.core``.
        """
        if self._owner is None:
            raise ValueError("jno.peec: this solution carries no per-conductor breakdown.")

        def _per_region(power, owner, vol, names):
            """`sum P / sum V` over each region's own elements, on the region's own mesh."""
            own, vol = np.asarray(owner), jnp.asarray(vol)
            out = {}
            for k, name in enumerate(names):
                sel = np.flatnonzero(own == k)
                if sel.size == 0:  # a region the discretisation gave no elements owns no loss
                    continue
                q = jnp.sum(power[:, sel], axis=1) / jnp.sum(vol[sel])
                out[name] = q[0] if jnp.ndim(self._port) == 0 else q
            return out

        cur = jnp.atleast_2d(self.i)
        channels = {
            "sigma": _per_region(jnp.einsum("k,fk->fk", self._R, jnp.abs(cur) ** 2), self._owner, self._vol, self._names)
        }
        if self._mag_cur is not None:
            # `w Im(R_m) |I_m|^2`, and every part of that spelling is load-bearing.
            #
            # IMAGINARY, not real: a reluctance is `l / (mu0 chi A)`, so a lossless core has a real
            # chi and a real R_m, and this channel is then identically zero -- which is the physics.
            # The lossy component of `mu_r` is what puts an imaginary part in chi, and `1/chi` turns
            # it into the imaginary part of the reluctance.
            #
            # And times OMEGA: R_m is a reluctance, so `R_m |I_m|^2` is not a power at all. The
            # first version of this readout omitted the factor and took the real part, and it gave
            # 1.36e-06 W where the port was delivering 8.58e-02 W more than the conductors were
            # dissipating -- a number with the right shape, the right sign and no relation to the
            # energy. The power balance below is what caught it.
            w = 2 * np.pi * jnp.atleast_1d(jnp.asarray(self.freq))[:, None]
            pm = w * jnp.einsum("k,fk->fk", jnp.imag(self._mag_res), jnp.abs(jnp.atleast_2d(self._mag_cur)) ** 2)
            channels["mu_r"] = _per_region(pm, self._mag_owner, self._mag_vol, self._mag_names)
        return Dissipation(channels)

    def field(self, points):
        """Magnetic flux density at ``points``, tesla — ``(n, 3)``, or ``(n_freq, n, 3)`` swept.

        A partial-element method never meshes the air, so the field off the metal is not a solved
        unknown: it is the Biot-Savart sum over the currents that WERE solved for,

            B(r) = (mu0 / 4 pi) * sum_k I_k * (dl_k x (r - r_k)) / |r - r_k|^3

        evaluated on the same sub-point quadrature the operator uses, so a curved wire is integrated
        rather than treated as one straight segment. That makes it a READOUT: no second problem, no
        boundary condition, and differentiable in the currents and in ``points`` alike -- which is
        what an EMI objective over a keep-out volume needs.

        Free space only. There is no magnetic material in this solver, so a nearby core would change
        the answer and is not represented; see the scope note on the module.

        Args:
            points: ``(n, 3)`` positions, in metres. May be traced.
        """
        pts = jnp.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"jno.peec: field() takes (n, 3) positions in metres, got shape {tuple(np.shape(points))}.")
        f = self._fil
        src, mom, grp = jnp.asarray(f.pos), jnp.asarray(f.mom), jnp.asarray(f.group)
        cur = jnp.atleast_2d(self.i)

        # The kernel is singular ON a filament, and the field INSIDE metal is not what this computes
        # either -- so a probe within a conductor's own cross-section is refused rather than answered.
        # The scale is that cross-section, not an absolute tolerance: a busbar and a bond wire are
        # both legal. (An on-AXIS point is not the dangerous case -- a straight filament's field
        # vanishes there by symmetry; it is being inside and OFF axis that diverges.)
        rad = np.sqrt(np.asarray(jax.lax.stop_gradient(jnp.asarray(f.area))) / np.pi)
        if not isinstance(pts, jax.core.Tracer):
            gap = np.linalg.norm(np.asarray(pts)[:, None, :] - np.asarray(jax.lax.stop_gradient(src))[None, :, :], axis=-1)
            r_of = np.repeat(rad, np.asarray(np.bincount(np.asarray(f.group), minlength=rad.size)))
            if np.any(gap < r_of[None, :]):
                bad = int(np.argmin(gap.min(axis=1)))
                raise ValueError(
                    f"jno.peec: field() was asked for a point {tuple(np.asarray(pts)[bad])} lying INSIDE "
                    "a conductor, where the Biot-Savart kernel is singular. This computes the free-space "
                    "field the solved currents produce; the field inside the metal is not it. Evaluate "
                    "off the conductor."
                )

        def _one(ik):
            m = mom * ik[grp][:, None]  # the moment already carries dl and the quadrature weight
            r = pts[:, None, :] - src[None, :, :]
            inv = jnp.linalg.norm(r, axis=-1) ** -3
            return (MU0 / (4 * jnp.pi)) * jnp.einsum("psi,ps->pi", jnp.cross(m[None, :, :], r), inv)

        out = jnp.stack([_one(jnp.real(c)) + 1j * _one(jnp.imag(c)) for c in cur])
        out = jnp.real(out) if not jnp.iscomplexobj(self.i) else out
        return out if jnp.ndim(self._port) else out[0]

    def export_vtk(self, save_path: str = "./runs/peec.vtk", freq_index: int | None = None):
        """Write the solved currents to a file a viewer can open — the same verb as
        :meth:`jno.domain.export_vtk`, and the same job.

        A partial-element network is a set of straight segments, so it exports as one: each filament
        a ``line`` cell between its two nodes, carrying its current magnitude, its phase, and the
        current DENSITY (which is what tells a crowded corner from a wide one, since the elements are
        not all the same size).

        Args:
            save_path: where to write. The extension picks the format, as meshio's do.
            freq_index: which point of a swept solve to write. Required when ``freq`` was an array —
                there is no single current field then, and picking one silently would be a guess.
        """
        import meshio

        cur = np.asarray(jnp.atleast_2d(self.i))
        if cur.shape[0] > 1 and freq_index is None:
            raise ValueError(
                f"jno.peec: this solution holds {cur.shape[0]} frequencies, so export_vtk cannot tell "
                "which frequency to write. Say which: `export_vtk(path, freq_index=0)`."
            )
        ik = cur[0 if freq_index is None else int(freq_index)]

        f = self._fil
        nodes = np.asarray(f.nodes)
        inc = f.incidence.tocsc()
        # A filament's two ends are the two nonzeros of its incidence column: +1 leaves, -1 enters.
        ends = np.zeros((inc.shape[1], 2), dtype=int)
        for k in range(inc.shape[1]):
            rows = inc.indices[inc.indptr[k] : inc.indptr[k + 1]]
            vals = inc.data[inc.indptr[k] : inc.indptr[k + 1]]
            ends[k] = (rows[np.argmax(vals)], rows[np.argmin(vals)])
        area = np.asarray(jax.lax.stop_gradient(jnp.asarray(f.area)))
        data = {
            "current": [np.abs(ik)],
            "current_density": [np.abs(ik) / np.maximum(area, 1e-300)],
            "phase_deg": [np.degrees(np.angle(ik))],
        }
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        meshio.write_points_cells(save_path, nodes, [("line", ends)], cell_data=data)
        return save_path

    def current(self, terminal):
        """Net current injected at ``terminal``, amp."""
        if terminal not in self._terms:
            raise ValueError(f"jno.peec: no terminal {terminal!r}. Known: {sorted(self._terms)}.")
        return self._terms[terminal]

    def voltage(self, a, b=None):
        """Potential at terminal ``a``, or the difference ``a - b``, volt.

        This is the readout an OPEN terminal needs. A transformer's unloaded secondary carries no
        current, so ``current`` and the port impedance say nothing about it -- the induced voltage
        is in the nodal potentials alone, and it is what a turns ratio is measured from.

        Prefer the TWO-terminal form. A single potential is only defined against whatever the solve
        pinned -- a declared ground, or the source's negative side when there is exactly one source
        and no ground -- so it moves if that reference changes, while a difference does not.

        A terminal's potential is its pad's, exactly as the source and device rows define it: the
        pad's nodes are shorted to each other, so any of them carries it. A WEIGHTED terminal is
        not shorted, and is then the weighted sum over the pad -- the same unnormalised weights the
        constraint rows use, so ``voltage`` and the circuit agree by construction.
        """
        if self._pot is None:
            raise ValueError(
                "jno.peec: this solution carries no nodal potentials, so there is no voltage to "
                "read. They are kept by `BuiltPEEC.solve`; a solution built by hand has none."
            )
        for t in (a, b):
            if t is not None and t not in self._pads:
                raise ValueError(f"jno.peec: no terminal {t!r}. Known: {sorted(self._pads)}.")
        phi = jnp.atleast_2d(self._pot)

        def at(t):
            spec = self._pads[t]
            ids, w = spec if isinstance(spec, tuple) else (spec, None)
            ids = np.asarray(ids, dtype=int)
            if w is None:  # shorted pad: every node carries the same potential
                return phi[:, ids[0]]
            return phi[:, ids] @ jnp.asarray(w).reshape(-1).astype(complex)

        out = at(a) - (0.0 if b is None else at(b))
        return out if jnp.ndim(self._port) else out[0]


class PEEC:
    """A partial-element problem: built from the constraint list, solved by :meth:`solve`."""

    def __init__(self, constraints, freq=0.0):
        self.constraints = list(constraints)
        if not self.constraints:
            raise ValueError("jno.peec: no constraints. A network with nothing impressed on it carries no current.")
        self.freq = np.atleast_1d(np.asarray(freq, dtype=float))
        self._scalar_freq = np.ndim(freq) == 0
        self.domain = _domain_of(self.constraints)
        self.sources, self.currents, self.grounds, self.devices = port_spec(self.constraints)
        if len(self.sources) != 1:
            raise ValueError(
                f"jno.peec: {len(self.sources)} sources; the impedance readouts describe ONE port. Write "
                "exactly one `v(A) - v(B) - volts`, and express the rest as fixed potentials or currents."
            )

    def _discretise(self):
        regions = dict(getattr(self.domain, "_shape_regions", {}) or {})
        named = (
            {t for s in self.sources for t in s[:2]}
            | {t for t, _ in self.currents}
            | {t for t, _ in self.grounds}
            | {t for dv in self.devices for t in dv[:2]}  # a device's ends are terminals, not conductors
        )
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

        def _attached(name):
            try:
                return self.domain.attached(name)
            except KeyError:  # nothing declared it at all: the same story as declaring it patchily
                return {}

        # WHAT A REGION CARRIES DECIDES WHAT IT IS -- there is no mode flag and no solver argument.
        # `sigma` alone is a conductor; `mu_r` alone is a core that does not conduct, a ferrite or a
        # powder; both together is a conducting magnetic material, a lamination or a lossy core.
        sig, mur = _attached("sigma"), _attached("mu_r")
        lines, line_sig, solids = [], [], []
        line_names, solid_names = [], []
        magnetic, magnetic_names = [], []
        for n, sh in conductors.items():
            if n not in sig and n not in mur:
                raise ValueError(
                    f"jno.peec: region {n!r} declares neither a conductivity nor a permeability, so "
                    "there is nothing to solve on it. Give it one: `.attach(sigma=...)` for metal, "
                    "`.attach(mu_r=...)` for a core that does not conduct, or both for a conducting "
                    "magnetic material."
                )
            node = getattr(sh, "_node", None)
            kind = type(node[1]).__name__ if (isinstance(node, tuple) and node and node[0] == "leaf") else None
            if kind != "Line" and not getattr(sh, "is_analytic", lambda: False)():
                raise NotImplementedError(_shape_msg(n, kind))
            if n in mur:
                if kind == "Line":
                    raise NotImplementedError(
                        f"jno.peec: {n!r} attaches mu_r to a Shape.line. A core carries FLUX through a "
                        "cross-section, which a filament does not have -- model it as a solid."
                    )
                # What circulates in the magnetic mesh is the MAGNETISATION, whose constitutive
                # quantity is chi = mu_r - 1. Air adds none, so a mu_r of exactly 1 is not a magnetic
                # region at all -- it is dropped, which is exact rather than an approximation, and is
                # what lets a unit-permeability core reproduce the coreless answer to the last bit
                # instead of merely closely. It also keeps an infinite reluctance out of the solve.
                if _unit_permeability(mur[n]):
                    from .utils.solver.peec import _warn_once

                    _warn_once(
                        f"jno.peec: {n!r} attaches mu_r = 1, which is air: chi = mu_r - 1 is zero, so "
                        "it adds no magnetisation and is not discretised as a core. Nothing is lost -- "
                        "the answer is the one you would get without it -- but if a core was intended, "
                        "its permeability is not set."
                    )
                else:
                    magnetic.append((sh, mur[n]))
                    magnetic_names.append(n)
            if n not in sig:
                if n not in {m for m in magnetic_names}:
                    raise ValueError(
                        f"jno.peec: region {n!r} declares mu_r = 1 and no conductivity, so it is air "
                        "with a name. Give it a real permeability, give it a `sigma`, or leave it out "
                        "of the network."
                    )
                continue
            # NOT coerced to float: a conductivity may be a traced value, which is what closes the
            # electro-thermal loop — sigma(T) falls as the conductor heats, and copper is about 31 %
            # more resistive at 100 C than at 20 C.
            if kind == "Line":
                lines.append(sh)
                line_sig.append(sig[n])
                line_names.append(n)
            else:
                # ANY closed-form solid, not just a box: the lattice covers its bounding box and a
                # mask says which cells are metal, so a cylinder, a sphere, a union or a CSG
                # difference all voxelise the same way and all stay one Toeplitz block.
                solids.append((sh, sig[n]))
                solid_names.append(n)

        parts, owners, blocks = [], [], []  # (Filaments, sigma), the shapes, and (names, resolver)
        if lines:
            # A WELDED network shares one near-field block, which is vectorised over a single
            # sub-point count. A bar samples its VOLUME (quad x quad_t^2 points) because a lattice
            # cell is a cube; a wire is thin and needs none of that, so it spends the matching count
            # on points along its own length, where they do help it. With no solids there is nothing
            # to match and a wire keeps the plain rule -- the count is a WELD constraint, not a
            # property of a filament.
            fl = line_filaments(lines, quad=_QUAD * _QUAD_T**2 if solids else _QUAD)
            # Each conductor's conductivity is resolved over ITS OWN filaments, so a field sees the
            # midpoints of the wire it belongs to and a per-element vector is that wire's own count.
            cen, fpart = element_centres(fl), np.asarray(fl.part)
            sel = [np.flatnonzero(fpart == i) for i in range(len(lines))]
            inv = np.empty(len(fpart), dtype=int)
            inv[np.concatenate(sel)] = np.arange(len(fpart))

            # A wire is one-dimensional, so an anisotropic sigma reaches it as `t . sigma . t` along
            # its own tangent rather than as a component -- see `resolve_sigma`.
            _tan = np.asarray(fl.mom)[:: max(1, np.asarray(fl.mom).shape[0] // len(np.asarray(fl.length)))]

            def line_resolve(vals, _s=sel, _i=inv, _c=cen, _n=tuple(line_names), _t=_tan):
                return jnp.concatenate(
                    [
                        jnp.asarray(resolve_sigma(v, _c[q], f"conductor {nm!r}", tangent=_t[q]))
                        for v, q, nm in zip(vals, _s, _n)
                    ]
                )[_i]

            blocks.append((tuple(line_names), line_resolve))
            parts.append((fl, line_resolve(line_sig)))
            owners.append(lines)
        if solids:
            # ONE grid for every solid. Separate lattices couple through a block that is not Toeplitz,
            # so a shared grid is what keeps the whole trace layer a single FFT: on the example module
            # that is ten coplanar traces of equal thickness, which is the case it fits exactly.
            shs = [sh for sh, _ in solids]
            sgs = [sg for _, sg in solids]
            # NOT passing the build frequency, so no sheet pairs are emitted. `bar_filaments(freq=)`
            # discretises a conductor thick against the skin depth as a current sheet per face, which
            # fixes a real defect (a return plane's thickness moving L where it cannot) -- but the
            # model is WRONG for a conductor carrying the loop current, and by more than the defect
            # it fixes. On a real power module, crossing the pairing threshold collapses the loop
            # inductance discontinuously:
            #
            #     50 kHz, unpaired   60.5 nH        80 kHz, paired   20.1 nH
            #
            # where the physical change over that range is nil. The unpaired arm is the right one:
            # it agrees with pypeec at 1 kHz (79.0 against 76.8 nH) and trends to its 51.3 nH at
            # 1 MHz. Inductance cannot be discontinuous in frequency, so the feature stays off the
            # front door until that is understood; the machinery and its tests are kept.
            fb = bar_filaments(shs, sigma=sgs, grid_shapes=[s for s, _ in magnetic])
            blocks.append((tuple(solid_names), fb.lattice["resolve"]))
            parts.append((fb, fb.lattice["sigma"]))
            owners.append(shs)

        attached = dict(zip(line_names, line_sig))
        attached.update({n: sg for n, (_sh, sg) in zip(solid_names, solids)})

        def resolve_all(overrides=None):
            """Per-filament conductivity in _weld's concatenation order. Safe to call under a trace.

            Every piece it closes over is concrete geometry, so calling it again costs nothing but
            the arithmetic -- which is what makes a design iteration cheap and a solve jittable.
            """
            vals = dict(attached)
            for n, v in (overrides or {}).items():
                if n not in vals:
                    raise ValueError(
                        f"jno.peec: sigma={{{n!r}: ...}} names no conductor of this network. Its "
                        f"conductors are {sorted(vals)}. A terminal is not a conductor."
                    )
                vals[n] = v
            return jnp.concatenate([jnp.asarray(fn([vals[n] for n in names])) for names, fn in blocks])

        # part index -> region name, in the order _weld renumbers them: lines, then solids
        fil, _sigma0 = _weld(parts, owners)
        if line_names and solid_names:
            # Mixing a Shape.line with a solid puts the whole network on the WELDED path, which is a
            # different and far more expensive solver than a plain lattice -- and nothing else in the
            # model announces that, so a single bond wire silently changes the complexity.
            from .utils.solver.peec import _warn_once

            _warn_once(
                f"jno.peec: this network is WELDED -- {len(line_names)} line conductor(s) joined to "
                f"{len(solid_names)} solid(s) -- so it solves on the welded path, not the lattice "
                "one. A lattice applies its partial inductance by FFT behind a diagonal "
                "preconditioner; a weld needs a dense cross block between the parts and a "
                "whole-system near-field factorisation. Measured: 27,533 welded elements take 31 s "
                "where 114,000 lattice bars take 1.2 s, and adding ONE wire to a 6,806-bar lattice "
                "took it from 0.19 s to 19.8 s before that preconditioner existed. This is what "
                "joining a bond wire to a trace layer costs, not a mistake -- but a model built from "
                "solids alone stays on the fast path, so it is worth knowing which one you are on."
            )
        # The magnetic mesh is structurally the SAME object as the electric one: a core is voxels and
        # the faces between them, exactly as a conductor is bars between cells. So `bar_filaments`
        # builds it unchanged, carrying the permeability where a conductivity would sit.
        mag = None
        if magnetic:
            if lines:
                raise NotImplementedError(
                    "jno.peec: a magnetic region together with a Shape.line is not supported yet. A "
                    "welded network already needs a cross block and a whole-system factorisation, and "
                    "adding a coupled magnetic system to that is untested -- so it is refused rather "
                    "than guessed at. Model the winding as a solid, or drop the core."
                )
            # both meshes must land on ONE grid, or the coupling between them is not Toeplitz
            # CHI, not mu_r. The lattice series-averages whatever it is handed along an element,
            # and what adds in series is the reluctance -- so it is 1/chi that has to be averaged,
            # which is the harmonic mean of chi. Handing it mu_r would average the wrong quantity,
            # silently and only where two different core materials touch.
            mag = bar_filaments(
                [s for s, _ in magnetic],
                sigma=[jnp.asarray(m) - 1.0 for _, m in magnetic],
                grid_shapes=[s for s, _ in solids],
            )
        return fil, terms, line_names + solid_names, resolve_all, mag, tuple(magnetic_names)

    def build(self) -> "BuiltPEEC":
        """Freeze the discretisation and return the jittable :class:`BuiltPEEC`.

        Everything structural -- which cells are metal, which nodes a pad owns, which filaments weld
        to which -- is decided on the HOST from concrete geometry, and none of it can run under a
        trace. ``.solve()`` does that work every call, so it is differentiable but not jittable, and
        it redoes the same host pass on every design iteration.

        ``build()`` is that pass, done once. What comes back solves in pure jax, so it jits, and its
        conductivity is still free to be traced -- which is the split
        :meth:`jno.precond.ams.build` and :class:`jno.trace.FemLinearSystem` already use::

            emag = jno.peec([v(*a) - v(*b) - 1.0], freq=1e6).build()   # host, once
            jax.jit(lambda s: emag.solve(sigma={"trace": s}).L)(sig)   # traced, every iteration

        **Scope, up front.** The geometry is frozen, so a design variable may change a conductivity
        but not a shape: move a conductor, change a pitch or re-route a wire and the answer is for
        the geometry you built, silently. Build again for a new one. The thickness runs behind the
        skin term are frozen with it, so overriding a conductivity does not re-label which cells
        count as one material.
        """
        return BuiltPEEC(self, *self._discretise())

    def solve(self):
        """Solve at every frequency and return a :class:`PEECSolution`.

        Discretises and solves in one call. For a design loop use :meth:`build` instead and solve
        the result -- same answer, the host pass done once, and jittable.
        """
        return self.build().solve()


class BuiltPEEC:
    """A PEEC network with its discretisation frozen: the jittable, differentiable half.

    Comes from :meth:`peec.build`. Holds the filaments, the terminal node sets and the conductivity
    resolvers -- all of it concrete -- so :meth:`solve` touches nothing but jax and can be put
    inside ``jax.jit``, ``jax.grad``, or a ``jno.fn`` term driven by ``jno.core``.
    """

    def __init__(self, spec, fil, terms, part_names, resolve, mag=None, mag_names=()):
        self.fil, self.terminals, self.part_names = fil, tuple(terms), tuple(part_names)
        #: the magnetic mesh, when any region attached a `mu_r` -- a Filaments of the same shape as
        #: the electric one, since a core is voxels and faces just as a conductor is.
        self.mag, self.mag_names = mag, tuple(mag_names)
        self.freq, self._scalar_freq = spec.freq, spec._scalar_freq
        self.sources, self.currents, self.grounds, self.devices = (
            spec.sources,
            spec.currents,
            spec.grounds,
            spec.devices,
        )
        self._resolve = resolve
        # Which nodes each pad owns is STRUCTURAL -- `terminal_nodes` reads coordinates, which are
        # tracers under a gradient -- so it is resolved once here, on the reference geometry.
        self.nodes = {t: terminal_nodes(fil, sh) for t, sh in terms.items()}
        # The through-thickness guard, here rather than per solve: it reads the conductivity, which
        # inside a jit has no value. The declared one is the conservative case (see the note there),
        # and the geometry it also reads is frozen, so once is both enough and the only chance.
        from .utils.solver.peec import _check_unresolved_thickness

        sig0 = self._resolve(None)
        for f in self.freq:
            _check_unresolved_thickness(fil, sig0, 2 * np.pi * float(f), MU0)

    def solve(self, sigma=None, devices=None, weights=None, restart=None, matrix_free=None, operator=None):
        """Solve at every frequency and return a :class:`PEECSolution`.

        Args:
            sigma: optional ``{conductor: value}`` overriding what the geometry declared, in any
                spelling ``.attach(sigma=...)`` takes -- a scalar, a callable of position, or a
                vector per element. This is where a traced design variable enters a jitted loop:
                the shape captured at build time cannot close over a tracer that does not exist
                yet, so the value is handed in at solve instead.
            devices: optional ``{terminal: Z}`` overriding a two-terminal device's impedance, keyed
                by the terminal its constraint was written on. The same story as ``sigma`` and for
                the same reason: a device value that DEPENDS on the solved state cannot be a
                constant in the constraint list. A SiC die's on-resistance rises about 0.5 %/K, so
                an electro-thermal fixed point has to re-impress it every pass -- and it is the
                dominant feedback, since the dies carry most of the loss and only theirs runs away.
            weights: optional ``{terminal: w}`` making a terminal a prescribed current DISTRIBUTION
                over its nodes instead of a short across them -- one weight per node of its support.

                This is what makes a terminal's POSITION a design variable. Unweighted, a pad is an
                equipotential node SET, and which nodes are in the set is a step function of where
                the pad is: sliding a die a quarter of a millimetre changes the answer by nothing at
                all, and then by 8 % when a node crosses the boundary. Weighted, the support is a
                frozen superset covering the travel and the weights are smooth in the position, so
                the gradient exists -- the same structure-frozen, values-traced split as ``sigma``.
            restart: GMRES restart depth on the matrix-free path. The default of 16 is where the
                curve flattens for a few thousand elements, and it is not where it flattens for
                twenty thousand: on the example module at a 0.7 mm pitch (21,980 bars) it leaves a
                9.3e-06 residual where 1e-6 is wanted, while 48 converges -- and finishes SOONER,
                176 s against the 290 s the shallower one spends failing. Raise it when the solve
                refuses; a deeper restart holds ``restart`` more vectors, which is megabytes.
            matrix_free: ``None`` decides by structure -- a network containing a bar lattice is
                applied by FFT, anything else forms the dense operator. ``False`` forces the dense
                path, exact but O(N^2) memory, so for small networks only.
            operator: ``jno.solve.hierarchical(...)`` to compress a WELDED network's dense blocks --
                the cross coupling between parts and a non-lattice part's own partial inductance.
                Opt in: without it the operator is exact, so no existing answer moves. **It does
                not yet pay on a bar lattice** -- ACA fails on the structural zeros between
                perpendicular bar families, those blocks are detected and stored densely, and the
                measured compression is 1.00x. See :mod:`~jno.utils.solver.hmatrix`.
        """
        fil, nodes, terms = self.fil, self.nodes, self.terminals
        if weights:
            nodes = dict(nodes)
            for name, w in weights.items():
                if name not in nodes:
                    raise ValueError(
                        f"jno.peec: weights={{{name!r}: ...}} names no terminal of this network. "
                        f"Its terminals are {sorted(nodes)}."
                    )
                nodes[name] = (np.asarray(nodes[name]), w)
        sigma = self._resolve(sigma)
        dev = self.devices
        if devices:
            at = {d[0]: i for i, d in enumerate(dev)}
            dev = list(dev)
            for name, z in devices.items():
                if name not in at:
                    raise ValueError(
                        f"jno.peec: devices={{{name!r}: ...}} names no device of this network. A device is "
                        f"keyed by the terminal its constraint was written on -- `v(A) - v(B) - Z*i(A)` is "
                        f"keyed 'A'. This network's devices are {sorted(at)}."
                    )
                a, b, _z = dev[at[name]]
                dev[at[name]] = (a, b, z)
            dev = tuple(dev)

        cur, port, drive, inject, link, pot, mcur = [], [], [], [], [], [], []
        for f in self.freq:
            c, phi, inj, im = solve_network(
                fil,
                sigma,
                nodes,
                self.sources,
                self.grounds,
                self.currents,
                dev,
                # the DECLARED impedances, always concrete, for the host-built preconditioner
                device_host={d[0]: d[2] for d in self.devices},
                omega=2 * np.pi * float(f),
                **({} if restart is None else {"restart": int(restart)}),
                **({} if matrix_free is None else {"matrix_free": bool(matrix_free)}),
                **({} if operator is None else {"operator": operator}),
                **({} if self.mag is None else {"mag": self.mag, "chi": self.mag.lattice["sigma"]}),
                magnetic_current=True,
            )
            cur.append(c)
            inject.append(inj)
            pot.append(phi)
            if im is not None:
                mcur.append(im)
            if self.mag is not None:
                # The TOTAL flux linkage, magnetisation included, read straight off the circuit
                # equation `Z_int I + j w (Lp I + K' I_m) = A' phi`. So it needs no magnetic unknown
                # of its own, and with no core it is `Lp I` to the last bit -- which is what
                # `test_the_two_inductance_forms_agree_without_a_core` pins.
                from .utils.solver.kernel import internal_impedance as _zint
                from .utils.solver.peec import _bcoo

                w_f = 2 * np.pi * float(f)
                z = _zint(fil.length, fil.area, fil.skin, fil.round_, w_f, sigma, MU0, fil.span)
                inc = fil.incidence.tocoo()
                # sparse, like everywhere else the incidence is applied: dense it is an
                # (elements x nodes) complex matrix, which on a real core model is gigabytes
                at = _bcoo(inc.data.astype(complex), inc.col, inc.row, inc.shape[::-1])
                link.append((at @ phi - z * c) / (1j * w_f))
            a, _b, g = self.sources[0]
            drive.append(inj[a])
            port.append(g / inj[a])
        from .utils.solver.kernel import internal_impedance

        # the SURFACE resistance, so the Joule readout matches the solve rather than a DC restatement
        w = 2 * np.pi * float(self.freq[0])
        res = jnp.real(internal_impedance(fil.length, fil.area, fil.skin, fil.round_, w, sigma, MU0, fil.span))

        cur = jnp.stack(cur)
        port = jnp.stack([jnp.asarray(p) for p in port])
        mag_break = (None, None, None, None, ())
        if mcur:
            from .utils.solver.kernel import magnetic_reluctance

            mc = jnp.stack(mcur)
            chi = self.mag.lattice["sigma"]
            mag_break = (
                mc[0] if self._scalar_freq else mc,
                magnetic_reluctance(self.mag.length, self.mag.area, jnp.asarray(chi) + 1.0, MU0),
                jnp.asarray(self.mag.area) * jnp.asarray(self.mag.length),
                np.asarray(self.mag.part),
                self.mag_names,
            )
        sol = PEECSolution(
            self.freq[0] if self._scalar_freq else self.freq,
            cur[0] if self._scalar_freq else cur,
            fil,
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
            link=(None if not link else (link[0] if self._scalar_freq else jnp.stack(link))),
            vol=jnp.asarray(fil.area) * jnp.asarray(fil.length),
            owner=np.asarray(fil.part),
            names=self.part_names,
            pot=(jnp.stack(pot)[0] if self._scalar_freq else jnp.stack(pot)),
            # the pads AFTER `weights=` was applied, so `voltage` reads the terminal the solve saw
            pads=nodes,
            mag=mag_break,
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
        f"jno.peec: conductor {name!r} ({kind or 'a CSG'} plan) has no closed-form membership, and the "
        "lattice is built by asking which cells lie inside it. Everything analytic works — a box, a "
        "cylinder, a sphere, a union, a difference — as does Shape.line, which keeps its exact "
        "cross-section instead of being voxelised. A swept or filleted solid does not."
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
    inc = sparse.block_diag([f.incidence for f in fils], format="csr")
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
        inc,
        cat("length"),
        cat("area"),
        jnp.asarray(nodes),
        shift("part"),
        cat("skin"),
        np.concatenate([np.asarray(f.round_) for f in fils]),
        np.concatenate([np.asarray(f.span) for f in fils]),
        # Each block keeps whatever structure it had — a lattice stays a lattice — and the blocks
        # couple through a cross term. That is what lets a trace layer stay an FFT while the bond
        # wires landing on it stay exact.
        {"welded": _spans(fils)},
        # Element pairings survive the weld. Dropping them left the two current sheets of a slab in
        # the network as INDEPENDENT elements -- each taking the whole conductor's surface impedance,
        # two of them in parallel, with none of the coupling that makes them one conductor. The
        # geometry still said "two sheets" while the physics said "two conductors", and the loop
        # inductance of a real module fell 3.3x the moment the pairing threshold was crossed.
        _pairs(fils),
    )
    # jnp, not numpy: a per-filament conductivity may be TRACED -- a density field, or sigma(T) --
    # and a welded network is exactly where that matters, since a real module is traces AND wires.
    return welded, jnp.concatenate([jnp.asarray(g) for _f, g in parts])


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
    # merging rows is a left-multiply by the (kept x all) membership matrix, which keeps it sparse
    pick = sparse.coo_matrix((np.ones(len(nodes)), (inverse, np.arange(len(nodes)))), shape=(len(keep), len(nodes))).tocsr()
    return (pick @ inc).tocsr(), nodes[keep]


def _pairs(fils):
    """The sheet pairings, renumbered into the WELDED network's element numbering.

    None when nothing anywhere is paired, which is what an unpaired network expects to see.
    """
    out, off, live = [], 0, False
    for f in fils:
        k = int(np.asarray(f.length).shape[0])
        pr = getattr(f, "pair", None)
        if pr is None:
            out.append(-np.ones(k, dtype=int))
        else:
            pr = np.asarray(pr, dtype=int)
            live = live or bool((pr >= 0).any())
            out.append(np.where(pr >= 0, pr + off, -1))  # welding stacks the blocks, so shift
        off += k
    return np.concatenate(out) if live else None


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
