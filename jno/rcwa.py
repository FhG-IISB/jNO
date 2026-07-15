"""Rigorous Coupled-Wave Analysis (RCWA / Fourier Modal Method) for periodic, layered structures.

RCWA is semi-analytic in the propagation direction: each layer is expanded in a truncated Fourier
basis in-plane and solved by an eigenmode decomposition, then layers are stitched with a scattering
matrix. For **periodic, piecewise-z-invariant** structures — extruded metasurface pillars are the
canonical case — this is far cheaper than a full 3-D volumetric solve.

Two entry points, both on `jno.rcwa`:

* **From a jNO problem (the front door).** Hand it the same constraint list you would give
  :func:`jno.fem` (or the already-built ``FEM``) — nothing else is required::

      rc  = jno.rcwa(constraints, orders=300)
      sol = rc.solve()                     # everything inferred from the traced problem
      rc.spec                              # the inferred RcwaSpec (available without fmmax)

  Inferred from the problem: **periodicity + period** (from the Floquet ties — absent ⇒ raise, never
  assumed), the **super/substrate ambients** (the z-normal radiation faces), the **permittivity**
  (the ``K0**2*eps`` coefficient recovered from the scalar Helmholtz volume term, sampled along z and
  grouped by :func:`detect_layers`), the **wavelength** (``k0`` from the vacuum superstrate, or pass
  ``wavelength=`` to override), and the **incident wave** (illuminated face + transverse angle
  ``k_in``, read from the assembled forcing).

* **Explicit layers (the backend).** :class:`Rcwa` takes a hand-built ``[(thickness, eps), ...]`` stack;
  this is what the front door constructs internally.

Built on `fmmax` (a differentiable JAX Fourier Modal Method), imported lazily so the core install stays
lean. Enable with the ``rcwa`` extra::

    pip install jax-neural-operators[rcwa]      # or: pixi run -e rcwa ...

It **never fails silently**: every inference (periodicity, layer invariance, order/grid Nyquist,
wavelength, energy balance, a recognisable source) is validated and raises :class:`RcwaError` with a
concrete fix. Two `fmmax` conventions are baked in: the Poynting flux is summed *with sign* (an
``abs()`` sum over-counts and yields ``T+R>1``), and ``JONES_DIRECT_FOURIER`` is the default
factorization (the naive rule converges poorly for high-contrast dielectrics such as a-Si).

Beyond a scalar ε, the front door also infers a **tensor permittivity** ``ε̂`` (from ``inner(ε̂ @ u, v)``)
and an **in-plane PML** (a complex coordinate stretch ``S = 1 + iσ/k`` written into the stiffness
coefficients of the scalar Helmholtz term). A uniaxial PML is exactly a diagonal Maxwell ``ε̂`` and ``μ̂``
(``ε̂ = ε·Λ``, ``μ̂ = Λ``, ``Λ = diag(SᵧS_z/Sₓ, SₓS_z/Sᵧ, SₓSᵧ/S_z)``), so the traced stretch is honoured
exactly and solved via `fmmax`'s general anisotropic eigensolve — turning a periodic supercell into an
isolated scatterer (the absorbing frame decouples the walls).

An **internal source** — a dipole / Gaussian emitter — is authored the same way you would drive `jno.fem`
or a PINN: a forcing term in the residual, ``- f·v`` (scalar) or ``- inner(J, v)`` (vector current). The
front door detects the trial-free/test-present volume summand, localizes it (centroid → point vs Gaussian
width; which z-layer), splits the stack at the source plane, and drives `fmmax`'s ``amplitudes_for_source``.
Then ``sol.power("up"|"down")`` and ``sol.extraction(...)`` give the emitted power and directionality (LED /
Purcell physics) instead of plane-wave ``T``/``R``.

References:
 * M. G. Moharam & T. K. Gaylord, "Rigorous coupled-wave analysis of planar-grating diffraction",
   J. Opt. Soc. Am. 71, 811 (1981).
 * M. G. Moharam, D. A. Pommet, E. B. Grann & T. K. Gaylord, "Stable implementation of the enhanced
   transmittance matrix approach", J. Opt. Soc. Am. A 12, 1077 (1995).
 * W. C. Chew & W. H. Weedon, "A 3D perfectly matched medium from modified Maxwell's equations with
   stretched coordinates", Microw. Opt. Technol. Lett. 7, 599 (1994) — the complex coordinate stretch.
 * Z. S. Sacks, D. M. Kingsland, R. Lee & J.-F. Lee, "A perfectly matched anisotropic absorber for use
   as an absorbing boundary condition", IEEE Trans. Antennas Propag. 43, 1460 (1995) — the uniaxial ε̂/μ̂ PML.
 * A. Taflove et al. (eds.), "Advances in FDTD Computational Electrodynamics", Artech House (2013), ch. on
   modal dipole sources in periodic media — the internal-source (``amplitudes_for_source``) formulation.
 * fmmax: https://github.com/facebookresearch/fmmax
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np


def _concrete(x):
    """True when ``x`` is a real (non-traced) value, so a never-silent guard may inspect it. Under
    jit/grad the value is a tracer -- the guard steps aside (validation ran on the eager forward pass)."""
    return not isinstance(x, jax.core.Tracer)


_FMMAX_HINT = (
    "jno.rcwa needs the optional fmmax backend: pip install jax-neural-operators[rcwa] "
    "(or `pixi run -e rcwa ...`, or `pip install fmmax`)."
)


def _fmmax():
    try:
        import fmmax

        return fmmax
    except ImportError as e:  # never a silent no-op
        raise ImportError(_FMMAX_HINT) from e


class RcwaError(ValueError):
    """Raised for any ill-posed RCWA problem — always loud, never a silent wrong answer."""


# =====================================================================================
# Layer auto-detection: group z-invariant slabs of a sampled permittivity into RCWA layers.
# =====================================================================================
def detect_layers(E, z, tol=1e-3, slices=None):
    """Group a z-sampled permittivity into RCWA layers.

    Parameters
    ----------
    E:
        ``(Nz, Ng, Ng)`` permittivity sampled on the unit-cell grid at each height ``z``.
    z:
        ``(Nz,)`` ascending heights.
    tol:
        In-plane change below which two adjacent z-slices are the same material.
    slices:
        If given, staircase a continuously-varying ``eps`` into this many layers instead of raising.

    Returns
    -------
    list[tuple[float, np.ndarray]]
        ``[(thickness, eps_xy), ...]`` super→substrate; the two ambients have thickness ``inf``.

    Raises
    ------
    RcwaError
        If ``eps`` varies continuously in z and ``slices`` is None (RCWA needs invariant slabs).
    """
    Nz = len(z)
    if E.shape[0] != Nz:
        raise RcwaError(f"detect_layers: E has {E.shape[0]} z-slices but z has {Nz}.")
    step = np.array([np.max(np.abs(E[k] - E[k - 1])) for k in range(1, Nz)])
    boundary = step > tol
    edges = [0] + [k for k in range(1, Nz) if boundary[k - 1]] + [Nz]
    slabs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    thin = sum((b - a) <= 2 for a, b in slabs)
    if slices is None and len(slabs) > 4 and thin > 0.3 * len(slabs):
        raise RcwaError(
            f"eps varies continuously in z ({thin}/{len(slabs)} slabs are 1-cell thin) -> not "
            f"layer-invariant. Pass slices=N to staircase it, or the geometry isn't RCWA-appropriate."
        )
    if slices is not None:
        idx = np.linspace(0, Nz - 1, slices + 1).round().astype(int)
        slabs = [(idx[i], idx[i + 1]) for i in range(len(idx) - 1)]

    layers, report, zmids, zspans = [], [], [], []
    for i, (a, b) in enumerate(slabs):
        mid = (a + b) // 2
        var = float(np.max(np.abs(E[a:b] - E[mid])))
        if var > tol and slices is None:
            raise RcwaError(
                f"slab z=[{z[a]:.3f},{z[min(b, Nz - 1)]:.3f}] is not z-invariant (var={var:.2g}); pass slices=."
            )
        thick = np.inf if (i == 0 or i == len(slabs) - 1) else float(z[b - 1] - z[a])
        eps_xy = E[mid]
        kind = "uniform" if np.ptp(eps_xy) < tol else "patterned"
        layers.append((thick, eps_xy))
        zmids.append(float(z[mid]))
        zspans.append((float(z[a]), float(z[min(b - 1, Nz - 1)])))  # (z_lo, z_hi) -> place an internal source
        report.append(
            f"  layer {i}: z=[{z[a]:.3f},{z[min(b - 1, Nz - 1)]:.3f}] {kind} eps~[{eps_xy.min():.2f},{eps_xy.max():.2f}]"
        )
    detect_layers.last_report = "detected layers:\n" + "\n".join(report)
    detect_layers.last_zmid = zmids  # representative z of each layer -- lets a param sweep re-sample eps
    detect_layers.last_zspan = zspans  # (z_lo, z_hi) per layer -- lets an internal source split its layer
    return layers


# =====================================================================================
# The forward engine (explicit layers) — the fmmax backend the front door constructs.
# =====================================================================================
class _Sol:
    def __init__(self, fm, s, layers, expansion, nt, Pin, wavelength, thick=None, period=None):
        self._fm, self._s, self._layers, self._ex = fm, s, layers, expansion
        self._nt, self._Pin, self._wl = nt, Pin, wavelength
        self._thick, self._period = thick, period
        self._fwd = np.zeros((2 * nt, 1), complex)

    def _flux(self, amps, layer, backward=False):
        zero = jnp.zeros_like(amps)
        f, b = self._fm.directional_poynting_flux(zero if backward else amps, amps if backward else zero, layer)
        return jnp.reshape(b if backward else f, (-1,))  # SIGNED — energy-correct; jax-native (differentiable)

    def efficiency(self, kind):
        """Total transmitted (``"T"``) or reflected (``"R"``) power fraction over propagating orders.

        Returns a JAX scalar -- differentiable in the design (permittivity) so ``jax.grad`` of a
        transmission/reflection objective flows through the modal solve."""
        if kind == "T":
            return jnp.sum(self._flux(self._s.s11 @ self._fwd, self._layers[-1])) / self._Pin
        if kind == "R":
            return -jnp.sum(self._flux(self._s.s21 @ self._fwd, self._layers[0], backward=True)) / self._Pin
        raise RcwaError(f"efficiency(kind): kind must be 'T' or 'R', got {kind!r}")

    def order(self, m, n):
        """Diffraction efficiency into transmitted order ``(m, n)`` (raises if outside the truncation).
        Returns a differentiable JAX scalar."""
        bc = np.asarray(self._ex.basis_coefficients)  # static (truncation geometry) -> plain numpy index
        hit = np.where((bc[:, 0] == m) & (bc[:, 1] == n))[0]
        if len(hit) == 0:
            raise RcwaError(f"order ({m},{n}) is outside the truncation ({self._nt} terms); raise orders=.")
        i = int(hit[0])
        tf = jnp.abs(
            jnp.reshape(
                self._fm.directional_poynting_flux(
                    self._s.s11 @ self._fwd, jnp.zeros((2 * self._nt, 1), complex), self._layers[-1]
                )[0],
                (-1,),
            )
        )
        return (tf[i] + tf[i + self._nt]) / self._Pin

    def jones(self, kind="T"):
        """The 2×2 complex **Jones matrix** at the 0th diffraction order — how the structure maps incident
        polarization to transmitted (``"T"``) or reflected (``"R"``) polarization.

        ``J[q, p]`` is the 0th-order field amplitude in output polarization ``q`` for a unit-**power**
        incident wave in input polarization ``p``, normalised so ``|J[q, p]|²`` is the (co- or cross-
        polarised) power fraction and ``arg(J)`` carries the phase (a waveplate's retardation lives in the
        phase difference between the two diagonal entries). The columns sum in power to the total efficiency:
        ``sum_q |J[q, p]|² == efficiency(kind)`` for input ``p``. The two polarizations are fmmax's transverse
        Jones basis — for the 0th order at normal incidence they are the two in-plane axes (≈ x, y); an
        isotropic stack gives a diagonal ``J`` (no conversion), an in-plane-anisotropic one an off-diagonal
        ``J`` (polarization conversion). Differentiable in the design. Returns a JAX ``(2, 2)`` array."""
        fm, nt = self._fm, self._nt
        if kind == "T":
            smat, out_layer = self._s.s11, self._layers[-1]
        elif kind == "R":
            smat, out_layer = self._s.s21, self._layers[0]
        else:
            raise RcwaError(f"jones(kind): kind must be 'T' or 'R', got {kind!r}")
        flux_in = jnp.abs(jnp.real(jnp.reshape(fm.eigenmode_poynting_flux(self._layers[0]), (-1,))))
        flux_out = jnp.abs(jnp.real(jnp.reshape(fm.eigenmode_poynting_flux(out_layer), (-1,))))
        # fmmax orders eigenmodes by eigenvalue, not by Fourier order — the 0th order's two polarizations are
        # the two most-forward (largest-flux) ambient modes. The ambient is design-independent, so the two
        # indices are fixed geometry (read eagerly); the amplitudes/normalisation stay JAX (differentiable).
        order = np.argsort(-np.asarray(flux_in))
        pa, pb = int(order[0]), int(order[1])
        if _concrete(flux_in) and float(flux_in[pb]) <= 1e-9 * float(flux_in[pa] + 1e-30):
            raise RcwaError("only one forward-propagating polarization at the 0th order; the Jones matrix is degenerate.")

        def col(p):  # transmitted/reflected 0th-order amplitudes for a unit-power input in polarization p
            amp = jnp.reshape(smat @ jnp.zeros((2 * nt, 1), complex).at[p, 0].set(1.0), (-1,))
            return jnp.stack([amp[pa] * jnp.sqrt(flux_out[pa] / flux_in[p]), amp[pb] * jnp.sqrt(flux_out[pb] / flux_in[p])])

        return jnp.stack([col(pa), col(pb)], axis=1)  # (out q, in p)

    def field(self, y_frac=0.5, nx=80, density=40.0):
        """Reconstruct the real-space electric field on a vertical (x–z) slice at ``y = y_frac·Py``.

        Returns ``(intensity, extent, layer_z)``: ``intensity`` = ``|E|²`` of shape ``(nz, nx)`` (z
        vertical, x horizontal) for ``imshow``; ``extent`` = ``[0, Px, z_min, z_max]``; ``layer_z`` = the
        z-interfaces between layers (to annotate the patterned slab). Needs the finite ambient
        thickness recorded at solve time (semi-infinite ambients are shown as unit-thick slabs)."""
        fm = self._fm
        if self._thick is None:
            raise RcwaError("field() needs the layer thicknesses; call .solve() (which records them).")
        znum = [max(4, int(round(float(t) * density))) for t in self._thick]
        smi = fm.stack_s_matrices_interior(self._layers, self._thick)
        amps = fm.stack_amplitudes_interior(smi, self._fwd, np.zeros_like(self._fwd))
        efield, _h, (x, y, z) = fm.stack_fields_3d(amps, self._layers, self._thick, znum, grid_shape=(nx, nx))
        E = np.asarray(efield)  # (3, nx, ny, nz, 1)
        inten = np.sum(np.abs(E) ** 2, axis=0)[..., 0]  # (nx, ny, nz)
        yv = np.asarray(y)
        j = int(np.argmin(np.abs(yv[0, :] - y_frac * self._period[1])))
        slab = inten[:, j, :].T  # (nz, nx): z vertical, x horizontal
        zc = np.asarray(z).reshape(-1)
        layer_z = list(np.cumsum([float(t) for t in self._thick]))[:-1]
        return slab, [0.0, float(self._period[0]), float(zc.min()), float(zc.max())], layer_z


class _EmitterSol:
    """Readouts for an internal-source (dipole / Gaussian) emission solve: the power radiated **up** (into
    the superstrate) and **down** (into the substrate), and the **extraction** fraction into either side.

    ``power`` is in the source's own (un-normalised) units, so it scales with the source amplitude² -- useful
    as a relative objective and differentiable in the amplitude / orientation / design ε. ``extraction`` is
    the scale-free fraction ``up/(up+down)`` (or ``down/…``) -- the LED/emitter directionality figure of
    merit. (Purcell / LDOS -- total emitted power ÷ a homogeneous-medium reference -- is future work.)"""

    def __init__(self, up, down):
        self._up, self._down = up, down

    def power(self, kind="total"):
        if kind == "up":
            return self._up
        if kind == "down":
            return self._down
        if kind == "total":
            return self._up + self._down
        raise RcwaError(f"power(kind): kind must be 'up' | 'down' | 'total', got {kind!r}")

    def extraction(self, kind="up"):
        total = self._up + self._down
        if kind == "up":
            return self._up / total
        if kind == "down":
            return self._down / total
        raise RcwaError(f"extraction(kind): kind must be 'up' | 'down', got {kind!r}")


class Rcwa:
    """A periodic layered RCWA problem with an **explicit** layer stack (the backend engine).

    Most users reach this through :func:`jno.rcwa` with a problem list; construct it directly only when
    you already have the ``[(thickness, eps), ...]`` stack, period and truncation in hand.
    """

    def __init__(
        self,
        layers,
        *,
        period,
        orders,
        wavelength=None,
        k_in=(0.0, 0.0),
        formulation="JONES_DIRECT_FOURIER",
        assume_periodic=False,
    ):
        fm = _fmmax()
        if period is None or len(period) != 2 or period[0] <= 0 or period[1] <= 0:
            raise RcwaError(f"period must be (Px,Py) > 0, got {period!r}.")
        if not assume_periodic:
            raise RcwaError(
                "RCWA is periodic in-plane. If your cell is genuinely a period/supercell, pass "
                "assume_periodic=True. (The jno.rcwa front door sets this only after confirming the "
                "problem has Floquet ties.)"
            )
        if not layers:
            raise RcwaError("layers is empty: need at least [superstrate, ..., substrate].")
        for k, (t, e) in enumerate(layers):
            if not (t is None or (np.isreal(t) and t > 0) or t == np.inf):
                raise RcwaError(f"layer {k} thickness must be > 0 (or inf for ambient), got {t!r}.")
            eg = np.asarray(e)
            if eg.ndim == 2 and (eg.shape[0] < 4 or eg.shape[1] < 4):
                raise RcwaError(f"layer {k} eps grid {eg.shape} is too coarse to rasterize; sample it finer.")
        grids = [np.asarray(e).shape[0] for _, e in layers if np.asarray(e).ndim == 2]
        if grids:
            ng = min(grids)
            per_axis = int(np.ceil(np.sqrt(orders)))
            if ng < 2 * per_axis:
                raise RcwaError(
                    f"eps grid {ng}x{ng} under-resolves orders={orders} (~{per_axis}/axis); "
                    f"need >= {2 * per_axis} grid points/axis or fewer orders."
                )
        self.fm, self.layers_spec, self.period = fm, layers, period
        self.orders, self.wavelength, self.k_in = orders, wavelength, tuple(k_in)
        self.formulation = fm.Formulation[formulation]
        self.assume_periodic = True
        # Precompute the (design-INDEPENDENT) Fourier expansion ONCE, eagerly. fmmax's generate_expansion
        # mixes jnp with np.linalg.norm — fine eagerly, but under jax.jit jnp ops are tracers, so recomputing
        # it inside the traced solve raised TracerArrayConversionError. Cached here, the design solve is jit-safe.
        self._lv = fm.LatticeVectors(u=np.array([period[0], 0.0]), v=np.array([0.0, period[1]]))
        self._ex = fm.generate_expansion(self._lv, approximate_num_terms=orders)
        self._nt = self._ex.num_terms

    def _eigensolve_stack(self, layers_spec, wl, kin):
        """Eigensolve every layer (isotropic / anisotropic-ε / general ε&μ, by tuple length) and return the
        ``LayerSolveResult`` list plus the thickness list. Shared by the plane-wave and internal-source paths."""
        fm, lv, ex = self.fm, self._lv, self._ex

        def _grid(g):
            g = jnp.asarray(g) + 0j  # jax-native so a permittivity grid flows -> differentiable in ε
            return jnp.full((1, 1), g) if g.ndim == 0 else g

        def solve_layer(e):
            # general anisotropic layer: e = (ε_xx..ε_zz, μ_xx..μ_zz) -> ε AND μ tensors. Used for a uniaxial
            # PML (an in-plane coordinate stretch is a diagonal ε̂ and μ̂), and for magnetic / magneto-optic media.
            if isinstance(e, tuple) and len(e) == 10:
                exx, exy, eyx, eyy, ezz, uxx, uxy, uyx, uyy, uzz = (_grid(c) for c in e)
                return fm.eigensolve_general_anisotropic_media(
                    jnp.asarray(wl),
                    kin,
                    lv,
                    exx,
                    exy,
                    eyx,
                    eyy,
                    ezz,
                    uxx,
                    uxy,
                    uyx,
                    uyy,
                    uzz,
                    ex,
                    formulation=self.formulation,
                )
            # anisotropic layer: e = (ε_xx, ε_xy, ε_yx, ε_yy, ε_zz) grids -> fmmax's anisotropic eigensolve
            if isinstance(e, tuple) and len(e) == 5:
                exx, exy, eyx, eyy, ezz = (_grid(c) for c in e)
                return fm.eigensolve_anisotropic_media(
                    jnp.asarray(wl), kin, lv, exx, exy, eyx, eyy, ezz, ex, formulation=self.formulation
                )
            return fm.eigensolve_isotropic_media(jnp.asarray(wl), kin, lv, _grid(e), ex, formulation=self.formulation)

        layers = [solve_layer(e) for _, e in layers_spec]
        thick = [jnp.asarray(1.0 if t is None or t == np.inf else t) for t, _ in layers_spec]
        return layers, thick

    def solve(self, inc=None, wavelength=None, k_in=None, layers=None, source=None):
        """Solve the stack and return a :class:`_Sol`. Raises if the wavelength is unknown or energy is
        not conserved.

        ``layers`` optionally overrides the construction layer stack with ``[(thickness, eps), ...]`` at
        solve time -- pass JAX permittivity grids here to differentiate the solve in the design (the
        construction-time shape guards run once, eagerly, so the solve itself stays trace-clean).

        ``source`` (a dict built by the front door) drives an **internal-source** emission solve instead of
        plane-wave incidence -- see :meth:`_solve_source`; it returns an :class:`_EmitterSol`."""
        fm = self.fm
        wl = wavelength if wavelength is not None else self.wavelength
        if wl is None:
            raise RcwaError(
                "wavelength is unknown: pass wavelength= to solve() or at construction; it sets every "
                "layer's eigenmodes and is never defaulted."
            )
        layers_spec = self.layers_spec if layers is None else layers
        kin = jnp.asarray(self.k_in if k_in is None else k_in, float)  # jax -> incidence angle is differentiable
        nt = self._nt  # precomputed eagerly in __init__ (jit-safe)
        if source is not None:
            return self._solve_source(source, layers_spec, wl, kin)
        layers, thick = self._eigensolve_stack(layers_spec, wl, kin)
        s = fm.stack_s_matrix(layers, thick)
        # Incidence: excite the genuinely forward-propagating mode in the superstrate and normalise by
        # ITS flux. fmmax sorts eigenmodes by eigenvalue, so index 0 need not be the 0th forward order,
        # and a one-hot forward amplitude can carry zero forward flux -- pick the max-positive-flux mode.
        # (The superstrate is design-independent, so this is effectively constant, but stays jax-native so
        # the whole solve traces under grad/jit.)
        flux_sup = jnp.real(jnp.reshape(fm.eigenmode_poynting_flux(layers[0]), (-1,)))
        idx = jnp.argmax(flux_sup)
        Pin = flux_sup[idx]
        if _concrete(Pin) and float(Pin) <= 0:
            raise RcwaError("no forward-propagating incident mode in the superstrate; check wavelength/period.")
        fwd = jnp.zeros((2 * nt, 1), complex).at[idx, 0].set(1.0)
        sol = _Sol(fm, s, layers, self._ex, nt, Pin, wl, thick=thick, period=self.period)
        sol._fwd = fwd
        T, R = sol.efficiency("T"), sol.efficiency("R")
        if _concrete(T):  # never-silent runtime guards run on the eager forward pass; they step aside under trace
            if not (np.isfinite(float(T)) and np.isfinite(float(R))):
                raise RcwaError(
                    "non-finite efficiency (NaN/inf): likely a Rayleigh/Wood anomaly where a diffraction "
                    "order is exactly grazing (the wavelength coincides with a period resonance, e.g. "
                    f"wavelength == period). Nudge the wavelength or period. (period={self.period}, wl={wl})"
                )
            if float(T) + float(R) > 1.0 + 5e-3:
                raise RcwaError(
                    f"energy not conserved: T+R={float(T) + float(R):.4f} > 1 at orders={self.orders}. The modal "
                    f"solve is not converged (raise orders) or the structure is unphysical."
                )
        return sol

    def _solve_source(self, source, layers_spec, wl, kin):
        """Internal-source (dipole / Gaussian) emission solve. ``source`` is a dict from the front door::

            {x0, y0, layer, t_upper, t_lower, kind ('delta'|'gaussian'), fwhm, orient (ox,oy,oz), amp}

        The stack is split at the source plane (layer ``layer`` divided into ``t_upper`` above / ``t_lower``
        below the source), fmmax builds the dipole current ``(jx,jy,jz) = amp·orient·source_coeffs`` and
        ``amplitudes_for_source`` propagates it to the ambients. Returns an :class:`_EmitterSol` with the
        power radiated up / down (differentiable in ``amp``, ``orient`` and the design ε)."""
        fm, lv, ex = self.fm, self._lv, self._ex
        layers, thick = self._eigensolve_stack(layers_spec, wl, kin)
        si = source["layer"]
        tu, tl = jnp.asarray(source["t_upper"]), jnp.asarray(source["t_lower"])
        before = layers[: si + 1]
        after = layers[si:]
        before_t = [*thick[:si], tu]
        after_t = [tl, *thick[si + 1 :]]
        s_before = fm.stack_s_matrix(before, before_t)
        s_after = fm.stack_s_matrix(after, after_t)
        loc = jnp.asarray([[float(source["x0"]), float(source["y0"])]])
        if source["kind"] == "gaussian":
            base = fm.gaussian_source(jnp.asarray(float(source["fwhm"])), loc, kin, lv, ex)
        else:
            base = fm.dirac_delta_source(loc, kin, lv, ex)
        ox, oy, oz = source["orient"]
        amp = jnp.asarray(source["amp"]) + 0j
        jx, jy, jz = amp * ox * base, amp * oy * base, amp * oz * base
        bwd0, _, _, _, _, fN = fm.amplitudes_for_source(jx, jy, jz, s_before, s_after)
        # power emitted up = backward-going flux in the superstrate; down = forward-going flux in the substrate
        _, ub = fm.directional_poynting_flux(jnp.zeros_like(bwd0), bwd0, layers[0])
        df, _ = fm.directional_poynting_flux(fN, jnp.zeros_like(fN), layers[-1])
        up = jnp.abs(jnp.sum(jnp.real(ub)))
        down = jnp.abs(jnp.sum(jnp.real(df)))
        if _concrete(up) and not (np.isfinite(float(up)) and np.isfinite(float(down))):
            raise RcwaError(
                "non-finite emission (NaN/inf): the internal-source solve hit a Γ-point / Rayleigh degeneracy "
                f"(k_in={tuple(float(k) for k in kin)}). Nudge k_in off exact normal, e.g. k_in=(1e-3, 1e-3)."
            )
        return _EmitterSol(up, down)


# =====================================================================================
# Front door: infer an RCWA problem from a jNO constraint list / FEM problem + eps.
# =====================================================================================
@dataclass
class RcwaSpec:
    """Everything the RCWA engine needs, inferred from a jNO problem. Inspectable without fmmax."""

    period: tuple
    layers: list  # [(thickness, eps_xy), ...]
    wavelength: float
    k_in: tuple = (0.0, 0.0)
    source_face: str = ""
    ambient_faces: tuple = ()
    periodic_axes: dict = field(default_factory=dict)
    source: dict = None  # an internal-source (dipole/Gaussian) emission spec, else None (plane-wave incidence)

    def __repr__(self):
        return (
            f"RcwaSpec(period={tuple(round(p, 4) for p in self.period)}, layers={len(self.layers)}, "
            f"wavelength={self.wavelength}, k_in={tuple(round(k, 4) for k in self.k_in)}, "
            f"source_face={self.source_face!r})"
        )


def _walk_nodes(node):
    from jno._fem import _walk

    return _walk(node)


def _model_calls(node):
    return [nd for nd in _walk_nodes(node) if type(nd).__name__ == "ModelCall"]


def _as_fem(problem):
    """Accept a constraint list or a built FEM; return the FEM object."""
    from jno._fem import FEM
    from jno._fem import fem as _fem

    if isinstance(problem, FEM):
        return problem
    if isinstance(problem, (list, tuple)):
        return _fem(list(problem))
    raise RcwaError(
        "jno.rcwa(problem, ...): problem must be the constraint list you'd pass to jno.fem, or a built "
        f"FEM object; got {type(problem).__name__}."
    )


def _region_centroid(domain, tag):
    br = getattr(domain, "_boundary_regions", {}).get(tag)
    if br is None or len(getattr(br, "points", [])) == 0:
        raise RcwaError(f"boundary region {tag!r} has no points; cannot locate it.")
    return np.asarray(br.points).mean(0)


def _periodic_period(femobj, domain):
    """Recover {axis: (tagA, tagB)} and (Px, Py) from the Floquet ties. Raise if not periodic in x and y."""
    from jno._fem import _periodic_tie_spec

    ties = []
    for c in getattr(femobj, "_constraints", []):
        spec = _periodic_tie_spec(c, domain)
        if spec is not None:
            ties.append((spec[0], spec[1]))  # (master, slave) tags
    if not ties:
        raise RcwaError(
            "no Floquet/periodic ties found: RCWA solves an infinitely periodic cell. Author the "
            "in-plane boundaries as periodic ties (u(left)-u(right), u(front)-u(back)); a finite "
            "aperture with absorbing side walls is NOT periodic and cannot be solved by RCWA."
        )
    axes, period = {}, {}
    for a, b in ties:
        ca, cb = _region_centroid(domain, a), _region_centroid(domain, b)
        d = cb - ca
        ax = int(np.argmax(np.abs(d)))  # dominant separation axis: 0=x, 1=y
        axes["xyz"[ax]] = (a, b)
        period["xyz"[ax]] = float(abs(d[ax]))
    if "x" not in period or "y" not in period:
        raise RcwaError(
            f"RCWA needs periodicity in BOTH x and y; found ties only along {sorted(period)}. Add the missing periodic tie."
        )
    return axes, (period["x"], period["y"])


def _z_ambient_faces(domain):
    """The two z-normal radiation faces (bottom, top), by z-extent of the boundary-region centroids."""
    pts = np.asarray(domain.points)
    zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
    normals = getattr(domain, "normals_by_tag", {})
    bottom = top = None
    for tag, br in getattr(domain, "_boundary_regions", {}).items():
        if tag == "boundary" or br is None or len(getattr(br, "points", [])) == 0:
            continue
        nrm = np.asarray(normals.get(tag, np.zeros((1, 3)))).mean(0)
        if abs(nrm[2]) < 0.8 * (np.linalg.norm(nrm) + 1e-30):
            continue  # not a z-normal face
        cz = np.asarray(br.points)[:, 2].mean()
        if abs(cz - zmin) < abs(cz - zmax):
            bottom = tag
        else:
            top = tag
    if bottom is None or top is None:
        raise RcwaError(
            "could not identify the two z-normal ambient faces (superstrate/substrate). Tag the top and "
            "bottom faces of the cell so RCWA can place the ambients."
        )
    return bottom, top


def _param_values(femobj, eps):
    """{parameter_name: current value} for every trainable parameter the operator references. Uses the
    parameter's *initialized* value (not zeros) so a parameter that is a physical constant -- e.g. K0 in
    the source term -- doesn't vanish when the operator is assembled to read the source."""
    out = {}
    nodes = list(_model_calls(eps))
    for c in getattr(femobj, "_constraints", []):
        nodes += _model_calls(c)
    for mc in nodes:
        m = getattr(mc, "model", None)
        name = getattr(m, "_parameter_name", None)
        if name is None:
            continue
        out[name] = jnp.asarray(m.module.value)
    return out


def _eval_expr_nodes(node, domain):
    """Evaluate a coefficient expression at every mesh node (nodal parameters use their current value)."""
    import jax.numpy as jnp

    from jno.trace import Variable
    from jno.trace_evaluator import TraceEvaluator

    node_coords = jnp.asarray(np.asarray(domain.points))
    table = {mc.model.layer_id: mc.model.module for mc in _model_calls(node)}
    tags = {v.tag for v in _walk_nodes(node) if isinstance(v, Variable)}
    ctx = {t: node_coords for t in tags} if tags else {}
    return np.asarray(TraceEvaluator(table).evaluate(node, context=ctx)).reshape(-1)


# --- recover the permittivity coefficient (K0^2 * eps) from the volume weak form ------------------
def _add_split(node, sign=1, out=None):
    """Flatten a tree of +/- into signed additive summands."""
    from jno.trace import BinaryOp

    if out is None:
        out = []
    if isinstance(node, BinaryOp) and node.op in ("+", "-"):
        _add_split(node.left, sign, out)
        _add_split(node.right, sign if node.op == "+" else -sign, out)
    else:
        out.append((sign, node))
    return out


def _mul_factors(node, out=None):
    """Flatten a product tree into its multiplicative factors."""
    from jno.trace import BinaryOp

    if out is None:
        out = []
    if isinstance(node, BinaryOp) and node.op == "*":
        _mul_factors(node.left, out)
        _mul_factors(node.right, out)
    else:
        out.append(node)
    return out


def _volume_constraint(femobj):
    cls = list(getattr(femobj, "classification", []))
    cons = list(getattr(femobj, "_constraints", []))
    vols = [c for c, k in zip(cons, cls) if k == "volume"]
    if len(vols) != 1:
        raise RcwaError(
            f"expected exactly one volume weak-form term, found {len(vols)} (classification={cls}); "
            "RCWA infers the permittivity from a single scalar Helmholtz volume term."
        )
    return vols[0]


def _extract_permittivity_coeff(volume_term):
    """Pull the value-channel (mass) coefficient ``K0**2 * eps`` out of a Helmholtz / Maxwell volume term.

    SCALAR Helmholtz ``grad(u).grad(v) - K0**2 * eps * (u*v)``: the stiffness summands carry trial/test
    inside ``Jacobian`` nodes, the mass summand carries them as bare TrialFunction / TestFunction values.
    VECTOR Maxwell ``inner(curl u, curl v) - K0**2 * eps * inner(u, v)``: the curl-curl summand carries
    trial/test inside ``curl`` FunctionCalls, the mass summand as a bare ``inner(u, v)``. Either way we
    split additively, find the mass summand, and drop the trial/test factor(s) — what remains is
    ``K0**2 * eps`` (scalar ε only for now; anisotropic ε is a follow-on)."""
    import functools
    import operator

    from jno.trace import FunctionCall, TestFunction, TrialFunction

    def _bare_mass_inner(f):  # inner(u, v) with the trial & test *bare* (the vector Maxwell mass term)
        if not (isinstance(f, FunctionCall) and getattr(f, "_name", None) == "inner" and len(f.args) == 2):
            return False
        a0, a1 = f.args
        return (isinstance(a0, TrialFunction) and isinstance(a1, TestFunction)) or (
            isinstance(a0, TestFunction) and isinstance(a1, TrialFunction)
        )

    def _mass_matvec_matrix(f):  # inner(M @ u, v) with bare trial/test -> the tensor node M (anisotropic ε̂)
        if not (isinstance(f, FunctionCall) and getattr(f, "_name", None) == "inner" and len(f.args) == 2):
            return None

        def _mv(x):  # matvec(M, u) with a bare trial -> M
            if isinstance(x, FunctionCall) and getattr(x, "_name", None) == "matvec" and len(x.args) == 2:
                return x.args[0] if isinstance(x.args[1], TrialFunction) else None
            return None

        a0, a1 = f.args
        if (m := _mv(a0)) is not None and isinstance(a1, TestFunction):
            return m
        if (m := _mv(a1)) is not None and isinstance(a0, TestFunction):
            return m
        return None

    expr = getattr(volume_term, "expr", volume_term)
    mass = []
    for _sign, summand in _add_split(expr):
        fac = _mul_factors(summand)
        tv = [f for f in fac if isinstance(f, TrialFunction)]
        te = [f for f in fac if isinstance(f, TestFunction)]
        matvec_f = next((f for f in fac if _mass_matvec_matrix(f) is not None), None)
        if tv and te:  # scalar Helmholtz mass: bare u * v
            if any(getattr(f, "value_shape", ()) != () for f in tv + te):
                raise RcwaError(
                    "RCWA sees a bare vector trial×test product; author the Maxwell mass as inner(u, v) "
                    "(a vector curl-curl form) rather than a raw component product."
                )
            coeff_factors = [f for f in fac if not isinstance(f, (TrialFunction, TestFunction))]
        elif matvec_f is not None:  # anisotropic Maxwell mass: inner(ε̂ @ u, v) -> coeff = (scalars) · ε̂
            coeff_factors = [f for f in fac if f is not matvec_f] + [_mass_matvec_matrix(matvec_f)]
        elif any(_bare_mass_inner(f) for f in fac):  # vector Maxwell mass: inner(u, v)
            coeff_factors = [f for f in fac if not _bare_mass_inner(f)]
        else:
            continue  # stiffness / curl-curl / other channel
        if not coeff_factors:
            raise RcwaError("mass term has no coefficient factor; cannot recover permittivity.")
        mass.append(functools.reduce(operator.mul, coeff_factors))
    if not mass:
        raise RcwaError(
            "could not find a K0^2*eps mass term in the volume weak form (scalar u*v or vector inner(u,v)); "
            "RCWA needs a Helmholtz/Maxwell term. If ε is authored unusually, isolate it as a named coefficient."
        )
    return functools.reduce(operator.add, mass)


def _extract_pml_stretch(volume_term):
    """Recover an in-plane uniaxial PML coordinate-stretch from a scalar Helmholtz volume term.

    A PML writes the stiffness with anisotropic diagonal coefficients::

        c_xx·∂ₓu ∂ₓv + c_yy·∂ᵧu ∂ᵧv + c_zz·∂_zu ∂_zv

    where ``(c_xx, c_yy, c_zz) = Λ = diag(SᵧS_z/Sₓ, SₓS_z/Sᵧ, SₓSᵧ/S_z)`` is the coordinate-stretch
    tensor (``S_i = 1 + iσ_i/k``, ramping in the absorbing frame, ``1`` in the physical core). This is
    exactly the Maxwell uniaxial PML: ``ε̂ = ε·Λ``, ``μ̂ = Λ``. We honour whatever stretch the user traced.

    Returns ``{axis: Λ_axis_node}`` (axis ``0/1/2`` ↦ ``x/y/z``) when a stretch is present, or ``{}`` when
    every stiffness summand is a bare ``∂u·∂v`` (no PML). Raises on an off-diagonal stiffness
    (``∂ᵢu ∂ⱼv``, i≠j) — only a diagonal (uniaxial) stretch is supported."""
    import functools
    import operator

    from jno.trace import Jacobian

    expr = getattr(volume_term, "expr", volume_term)
    per_axis = {}  # axis -> list of coefficient nodes (None = a bare, coefficient-less ∂u·∂v)
    saw_stiffness = False
    for _sign, summand in _add_split(expr):
        fac = _mul_factors(summand)
        jt = [f for f in fac if isinstance(f, Jacobian) and _contains_trial(f.target)]
        je = [f for f in fac if isinstance(f, Jacobian) and _contains_test(f.target)]
        if not (jt and je):
            continue  # mass / source / other channel -- not a stiffness summand
        if len(jt) != 1 or len(je) != 1:
            raise RcwaError("RCWA PML: expected one trial-gradient and one test-gradient per stiffness term.")
        saw_stiffness = True
        axt, axe = jt[0].variables[0].dim[0], je[0].variables[0].dim[0]
        if axt != axe:
            raise RcwaError(
                "RCWA sees an off-diagonal stiffness term (∂ᵢu ∂ⱼv, i≠j) -> a non-diagonal coordinate "
                "stretch. Only a diagonal (uniaxial) in-plane PML is supported."
            )
        coeff_factors = [f for f in fac if f is not jt[0] and f is not je[0]]
        per_axis.setdefault(axt, []).append(functools.reduce(operator.mul, coeff_factors) if coeff_factors else None)
    if not saw_stiffness or all(c is None for cs in per_axis.values() for c in cs):
        return {}  # no stiffness, or every stiffness term is a bare ∂u·∂v -> not a PML
    return {  # a bare contribution on a stretched axis counts as Λ += 1
        ax: functools.reduce(operator.add, [(c if c is not None else 1.0) for c in cs]) for ax, cs in per_axis.items()
    }


def _pml_layer_components(mass_node, stretch, period, grid, zmid, params, k0sq):
    """The 10 relative-permittivity/permeability grids ``(ε_xx,ε_xy,ε_yx,ε_yy,ε_zz, μ_xx,μ_xy,μ_yx,μ_yy,μ_zz)``
    of an in-plane uniaxial PML layer at height ``zmid``.

    ``Λ = diag(c_xx, c_yy, c_zz)`` is read off the stiffness coefficients (``stretch``); the Maxwell PML is
    ``ε̂ = ε·Λ``, ``μ̂ = Λ`` with ``ε = (K0²·ε·SₓSᵧS_z)/k0² / (c_xx·c_yy·c_zz)`` since
    ``SₓSᵧS_z = c_xx·c_yy·c_zz``. Off-diagonal components are zero (uniaxial)."""
    pts = _cell_grid_at_z(period, grid, zmid)

    def samp(node):
        v = jnp.full((grid * grid,), node) if np.isscalar(node) else _eval_coeff_points(node, pts, params)
        return jnp.reshape(v, (grid, grid)) + 0j

    lam = [samp(stretch[ax]) for ax in (0, 1, 2)]
    m = jnp.reshape(_eval_coeff_points(mass_node, pts, params), (grid, grid)) + 0j
    eps_rel = (m / k0sq) / (lam[0] * lam[1] * lam[2])
    z = jnp.zeros((grid, grid), complex)
    return (eps_rel * lam[0], z, z, eps_rel * lam[1], eps_rel * lam[2], lam[0], z, z, lam[1], lam[2])


def _extract_volume_source(volume_term):
    """The **internal-source** forcing in the volume weak form: a summand that carries the TEST function but
    **no trial** — ``- f·v`` (scalar monopole) or ``- inner(J, v)`` (vector dipole, ``J`` a current density).

    Returns ``(source_node, is_vector)`` or ``None`` when the volume term is a pure operator (no forcing).
    The sign is dropped (emitted power ∝ |source|², sign-invariant); the node keeps any ``jno.np.parameter``
    it depends on, so a trainable source amplitude flows through the differentiable solve."""
    import functools
    import operator

    from jno.trace import FunctionCall, TestFunction, TrialFunction

    def _test_inner_other(f):  # inner(J, v) / inner(v, J) with a bare test v and J not a trial -> J
        if not (isinstance(f, FunctionCall) and getattr(f, "_name", None) == "inner" and len(f.args) == 2):
            return None
        a0, a1 = f.args
        if isinstance(a0, TestFunction) and not isinstance(a1, TrialFunction):
            return a1
        if isinstance(a1, TestFunction) and not isinstance(a0, TrialFunction):
            return a0
        return None

    expr = getattr(volume_term, "expr", volume_term)
    scal, vect = [], []
    for _sign, summand in _add_split(expr):
        if _contains_trial(summand) or not _contains_test(summand):
            continue  # operator term (has trial) or non-source
        fac = _mul_factors(summand)
        jinner = next((f for f in fac if _test_inner_other(f) is not None), None)
        if jinner is not None:  # vector source inner(J, v): source = (scalar factors)·J
            J = _test_inner_other(jinner)
            rest = [f for f in fac if f is not jinner]
            vect.append(functools.reduce(operator.mul, [*rest, J]) if rest else J)
        else:  # scalar source f·v: drop the bare test factor
            rest = [f for f in fac if not isinstance(f, TestFunction)]
            if rest:
                scal.append(functools.reduce(operator.mul, rest))
    if vect and scal:
        raise RcwaError("RCWA found both a scalar and a vector internal source in the volume term; author one.")
    if vect:
        return functools.reduce(operator.add, vect), True
    if scal:
        return functools.reduce(operator.add, scal), False
    return None


def _localize_source(src_node, is_vector, period, z_range, grid, nz, params):
    """Locate an internal source by its ``|·|²`` distribution on the cell volume: the centroid ``(x₀,y₀,z₀)``
    and lateral width → **point** (``dirac_delta_source``) vs **Gaussian** (``gaussian_source`` with a fwhm).
    Returns ``dict(x0, y0, z0, kind, fwhm)`` (all static geometry; the amplitude flows separately)."""
    xs = np.linspace(0, period[0], grid, endpoint=False)
    ys = np.linspace(0, period[1], grid, endpoint=False)
    zs = np.linspace(z_range[0], z_range[1], nz)
    GX, GY, GZ = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1)
    v = np.asarray(_eval_coeff_points(src_node, pts, params or {}))
    mag2 = np.sum(np.abs(v) ** 2, axis=-1) if (is_vector and v.ndim > 1) else np.abs(v).reshape(-1) ** 2
    if float(mag2.max()) <= 0:
        raise RcwaError("the internal source is identically zero at construction values; check its amplitude.")
    w = mag2 / mag2.sum()
    x0, y0, z0 = (float((w * pts[:, k]).sum()) for k in range(3))
    sx = float(np.sqrt(((pts[:, 0] - x0) ** 2 * w).sum()))
    sy = float(np.sqrt(((pts[:, 1] - y0) ** 2 * w).sum()))
    width = 0.5 * (sx + sy)  # lateral std; a point source is narrower than a cell
    dx = 0.5 * (period[0] + period[1]) / grid
    if width < 1.5 * dx:
        return dict(x0=x0, y0=y0, z0=z0, kind="delta", fwhm=0.0)
    return dict(x0=x0, y0=y0, z0=z0, kind="gaussian", fwhm=2.35482 * width)  # FWHM = 2.355·σ


def _has_nodal_field(node):
    """True if the coefficient depends on a per-node field parameter (needs mesh interpolation)."""
    return any(getattr(mc.model, "_fem_field", None) == "node" for mc in _model_calls(node))


def _contains_trial(node):
    from jno.trace import TrialFunction

    return any(isinstance(n, TrialFunction) for n in _walk_nodes(node))


def _contains_test(node):
    from jno.trace import TestFunction

    return any(isinstance(n, TestFunction) for n in _walk_nodes(node))


def _product_terms(factors, sign=1):
    """Expand a product of factors (some of which may be additive sums) into signed additive terms."""
    import functools
    import operator

    from jno.trace import BinaryOp

    for i, f in enumerate(factors):
        if isinstance(f, BinaryOp) and f.op in ("+", "-"):
            rest = factors[:i] + factors[i + 1 :]
            terms = []
            for s2, summ in _add_split(f):
                terms += _product_terms(rest + [summ], sign * s2)
            return terms
    return [(sign, functools.reduce(operator.mul, factors) if factors else 1.0)]


def _extract_source_coeff(term):
    """The trial-free (forcing) part of a boundary term ``coeff*v`` -- i.e. the incident wave. Peel the
    test factor, distribute the coefficient into additive terms, and keep those with no trial. Returns the
    source expression (an evaluable coefficient), or None if the term carries no forcing (e.g. a purely
    absorbing boundary ``-i k0 u v``)."""
    from jno.trace import FunctionCall, TestFunction

    def _vector_load_source(f):  # a factor `inner(g, n×v)` (trial-free g) -> the source g; else None
        if not (isinstance(f, FunctionCall) and getattr(f, "_name", None) == "inner" and len(f.args) == 2):
            return None
        a0, a1 = f.args

        def _cross_test(x):
            return isinstance(x, FunctionCall) and getattr(x, "_name", None) == "cross" and _contains_test(x)

        if _cross_test(a0) and not (_contains_trial(a1) or _contains_test(a1)):
            return a1
        if _cross_test(a1) and not (_contains_trial(a0) or _contains_test(a0)):
            return a0
        return None

    expr = getattr(term, "expr", term)
    parts = []
    for tsign, summand in _add_split(expr):
        fac = _mul_factors(summand)
        # vector Maxwell incident source: a factor inner(g, n×v); source = (other factors) · g
        vinner = next((f for f in fac if _vector_load_source(f) is not None), None)
        if vinner is not None:
            part = _vector_load_source(vinner)
            for cf in (f for f in fac if f is not vinner):
                part = cf * part
            parts.append((tsign, part))
            continue
        if not any(isinstance(f, TestFunction) for f in fac):
            continue
        rest = [f for f in fac if not isinstance(f, TestFunction)]
        for psign, part in _product_terms(rest, tsign):
            if not _contains_trial(part):
                parts.append((psign, part))
    if not parts:
        return None
    total = None
    for psign, part in parts:
        t = part if psign > 0 else (-1.0) * part
        total = t if total is None else total + t
    return total


def _kin_from_source(src_coeff, period, z0, params):
    """Transverse wavevector k_in = ∇⊥(phase) of the incident wave, read from the source coefficient by a
    small phase difference across the illuminated face -- differentiable in a source ANGLE parameter.

    Works for a scalar source (Helmholtz) and a VECTOR source (Maxwell, ``inner(g, n×v)``): a plane wave is
    ``E_inc·e^{i k⊥·r}``, so every component carries the same transverse phase — read it off the dominant
    component (avoids the N1E edge-sign corruption that a b-based read suffers)."""
    dx, dy = period[0] * 1e-3, period[1] * 1e-3
    x0, y0 = period[0] * 0.5, period[1] * 0.5

    def _ev(pt):
        return jnp.reshape(_eval_coeff_points(src_coeff, np.array([pt]), params), (-1,))

    base, sx, sy = _ev([x0, y0, z0]), _ev([x0 + dx, y0, z0]), _ev([x0, y0 + dy, z0])
    i = jnp.argmax(jnp.abs(base))  # dominant component (index 0 for a scalar source)
    kx = jnp.angle(sx[i] / base[i]) / dx
    ky = jnp.angle(sy[i] / base[i]) / dy
    return jnp.stack([kx, ky])


def _sample_grid_direct(coeff_node, grid, nz, period, z_range, params=None):
    """Evaluate an analytic coefficient expression directly on the RCWA grid (exact, no mesh noise).

    Used when the permittivity is an analytic function of the coordinates (no per-node field); this
    avoids the staircase/interp noise that makes z-slices look non-invariant on an unstructured mesh.
    ``params`` substitutes trainable-parameter values (so the eager structure/k0 inference uses valid
    values, e.g. a K0 parameter)."""
    xs = np.linspace(0, period[0], grid, endpoint=False)
    ys = np.linspace(0, period[1], grid, endpoint=False)
    zs = np.linspace(z_range[0], z_range[1], nz)
    GX, GY, GZ = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1)
    vals = np.asarray(_eval_coeff_points(coeff_node, pts, params or {}))
    if vals.ndim >= 3 and vals.shape[-2:] == (3, 3):  # anisotropic ε̂ -> layer-detect on the xx component
        vals = vals[..., 0, 0]
    vals = vals.reshape(grid, grid, nz)
    return np.moveaxis(vals, 2, 0).astype(complex), zs  # -> (nz, grid, grid)


def _coeff_is_tensor(coeff_node, period, z_range, params):
    """True if the permittivity coefficient is a 3×3 tensor (anisotropic ε̂) rather than a scalar — decided
    by evaluating it once at a representative cell point and checking the value shape."""
    pt = np.array([[period[0] * 0.5, period[1] * 0.5, 0.5 * (z_range[0] + z_range[1])]])
    v = np.asarray(_eval_coeff_points(coeff_node, pt, params or {}))
    return v.ndim >= 3 and v.shape[-2:] == (3, 3)


def _tensor_components(coeff_node, period, grid, zmid, params, k0sq):
    """The 5 relative-permittivity component grids (ε_xx, ε_xy, ε_yx, ε_yy, ε_zz) fmmax needs, sampled on the
    unit-cell grid at height ``zmid`` from the K0²·ε̂ tensor coefficient (divided by k0²)."""
    T = jnp.reshape(_eval_coeff_points(coeff_node, _cell_grid_at_z(period, grid, zmid), params), (grid, grid, 3, 3))
    T = T / k0sq
    return (T[..., 0, 0], T[..., 0, 1], T[..., 1, 0], T[..., 1, 1], T[..., 2, 2])


def _sample_grid(domain, eps_nodes, grid, nz, period, z_range):
    """Interpolate nodal eps onto an (nz, grid, grid) unit-cell array over the z-extent."""
    from scipy.interpolate import griddata

    pts = np.asarray(domain.points)
    xs = np.linspace(0, period[0], grid, endpoint=False)
    ys = np.linspace(0, period[1], grid, endpoint=False)
    zs = np.linspace(z_range[0], z_range[1], nz)
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    E = np.empty((nz, grid, grid), complex)
    for k, zz in enumerate(zs):
        sel = np.abs(pts[:, 2] - zz) < (z_range[1] - z_range[0]) / (nz - 1) * 0.75 + 1e-9
        if sel.sum() < 3:
            j = np.argsort(np.abs(pts[:, 2] - zz))[: max(3, grid)]
            sel = np.zeros(len(pts), bool)
            sel[j] = True
        vals = griddata(pts[sel][:, :2], eps_nodes[sel], (GX, GY), method="nearest")
        E[k] = vals
    return E, zs


def _source_dof_coords(domain, n_dof):
    """Coordinates the flat solution/forcing DOFs live on: mesh vertices for a nodal (Lagrange) field,
    or edge midpoints for a Nédélec (N1E) edge field, whose DOFs are edges — so the source-face detection
    reads the right positions for a vector Maxwell problem."""
    pts = np.asarray(domain.points)
    if n_dof == len(pts):
        return pts  # nodal (scalar Helmholtz)
    from jno.utils.solver.fem_topology import BASIX_TET_EDGES, BASIX_TRIANGLE_EDGES, build_edge_topology

    dim = getattr(domain, "dimension", 3)
    cells = np.asarray(domain.mesh.cells_dict["tetra" if dim == 3 else "triangle"])
    topo = build_edge_topology(cells, BASIX_TET_EDGES if dim == 3 else BASIX_TRIANGLE_EDGES)
    if topo.n_edges == n_dof:  # N1E edge DOFs -> edge midpoints
        return pts[np.asarray(topo.edge_vertices)].mean(axis=1)
    raise RcwaError(
        f"cannot map the {n_dof} forcing DOFs to coordinates (mesh has {len(pts)} vertices, "
        f"{topo.n_edges} edges); RCWA source-face detection supports nodal (Lagrange) and N1E fields."
    )


def _source_kin(femobj, domain, params=None):
    """Read the illuminated face and transverse wavevector k_in from the assembled forcing b.
    ``params`` overrides trainable-parameter values used when assembling (e.g. a K0/source parameter)."""
    op = femobj.operator
    if not (isinstance(op, tuple) and len(op) == 2):
        raise RcwaError(
            "jno.rcwa expects a complex (time-harmonic) Helmholtz problem: fem.operator did not return a "
            "(real, imag) operator pair. Author the field with complex=True."
        )
    opr, opi = op
    if hasattr(opr, "evaluate"):  # lazy (parametric) operator
        pz = _param_values(femobj, femobj._constraints[0] if femobj._constraints else 0)
        pz.update({k: jnp.asarray(v) for k, v in (params or {}).items()})
        _, br = opr.evaluate(pz)
        _, bi = opi.evaluate(pz)
    else:  # eager (parameter-free) (A, b) legs
        br, bi = opr[1], opi[1]
    b = np.asarray(br).reshape(-1) + 1j * np.asarray(bi).reshape(-1)
    nz = np.where(np.abs(b) > 1e-9 * (np.abs(b).max() + 1e-30))[0]
    if len(nz) == 0:
        raise RcwaError(
            "no source/forcing found in the problem: nothing to illuminate the cell with. Author an "
            "incident wave (e.g. an inhomogeneous absorbing boundary carrying the incoming field)."
        )
    pts = _source_dof_coords(domain, len(b))
    zc = pts[nz, 2]
    face_z = np.median(zc)
    if np.ptp(zc) > 1e-6 * (pts[:, 2].max() - pts[:, 2].min() + 1e-30) + 1e-9:
        raise RcwaError(
            f"the forcing is spread over a z-range [{zc.min():.3g},{zc.max():.3g}], not a single "
            f"illuminated face; RCWA needs the incident wave on one ambient face."
        )
    # transverse angle from the phase gradient of b across the lit face
    xy = pts[nz, :2]
    phase = np.unwrap(np.angle(b[nz]))
    A = np.c_[xy, np.ones(len(xy))]
    (kx, ky, _), *_ = np.linalg.lstsq(A, phase, rcond=None)
    k_in = (float(kx), float(ky))
    if abs(kx) < 1e-6 and abs(ky) < 1e-6:
        k_in = (0.0, 0.0)
    return face_z, k_in


def _cell_grid_at_z(period, grid, z):
    xs = np.linspace(0, period[0], grid, endpoint=False)
    ys = np.linspace(0, period[1], grid, endpoint=False)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), np.full(grid * grid, z)], 1)


def _bary_weights(node_coords, grid_pts):
    """Precompute, for each grid point, the 4 containing-tetrahedron node ids and their **barycentric**
    weights -- so ``rho_grid = sum_i w_i * rho[node_i]`` is a smooth, differentiable interpolation of the
    nodal design onto the RCWA grid (points outside the mesh hull fall back to the nearest node)."""
    from scipy.spatial import Delaunay, cKDTree

    tri = Delaunay(node_coords)
    m = len(grid_pts)
    simplex = tri.find_simplex(grid_pts)
    nodes = np.zeros((m, 4), np.int64)
    w = np.zeros((m, 4), np.float64)
    inside = simplex >= 0
    if inside.any():
        s = simplex[inside]
        T = tri.transform[s]  # (k, 4, 3): [:3] inverse affine, [3] the last-vertex offset
        b3 = np.einsum("kij,kj->ki", T[:, :3, :], grid_pts[inside] - T[:, 3, :])
        nodes[inside] = tri.simplices[s]
        w[inside] = np.c_[b3, 1.0 - b3.sum(axis=1)]
    if (~inside).any():  # outside the convex hull -> nearest node (weight 1)
        _, nn = cKDTree(node_coords).query(grid_pts[~inside])
        nodes[~inside] = nn[:, None]
        w[~inside, 0] = 1.0
    return nodes, w


def _eval_coeff_points(coeff_node, pts, params):
    """Evaluate the permittivity coefficient at ``pts``, substituting trainable parameters from
    ``params`` (jax values). Returns a JAX array -- differentiable in the design parameters."""
    import equinox as eqx

    from jno.trace import Variable
    from jno.trace_evaluator import TraceEvaluator

    table = {}
    for mc in _model_calls(coeff_node):
        mod, name = mc.model.module, getattr(mc.model, "_parameter_name", None)
        if name is not None and name in params:
            mod = eqx.tree_at(lambda m: m.value, mod, jnp.asarray(params[name]))
        table[mc.model.layer_id] = mod
    tags = {v.tag for v in _walk_nodes(coeff_node) if isinstance(v, Variable)}
    ctx = {t: jnp.asarray(pts) for t in tags} if tags else {}
    return TraceEvaluator(table).evaluate(coeff_node, context=ctx)


class _ParametricSol:
    """The solution of a parameterised RCWA problem. Its readouts are **trace nodes** over the graph's
    ``jno.np.parameter`` values (exactly like a parametric ``jno.fem`` solve), so ``rc.solve().efficiency``
    composes into a differentiable loss with **no solve arguments** -- the parameters live in the
    constraint list, and crux threads them. One traced layer, any solver."""

    def __init__(self, problem, rpe):
        self._p, self._rpe = problem, rpe

    def _node(self, method, *args):
        from jno.trace import FunctionCall

        self._p._engine()  # build the engine + its Fourier expansion EAGERLY here (outside any trace) so the
        # FunctionCall below reuses the cached engine and stays jit-safe under jax.jit(value_and_grad(...)).
        names = list(self._rpe)
        exprs = [self._rpe[n] for n in names]

        def fn(*values):  # crux threads the current parameter values in here
            return getattr(self._p._solve_at(dict(zip(names, values))), method)(*args)

        return FunctionCall(fn, exprs, name=f"rcwa_{method}")

    def efficiency(self, kind):
        return self._node("efficiency", kind)

    def order(self, m, n):
        return self._node("order", m, n)

    def power(self, kind="total"):
        return self._node("power", kind)

    def extraction(self, kind="up"):
        return self._node("extraction", kind)

    def field(self, *a, **k):
        raise RcwaError(
            "field() on a parameterised solve is not a scalar readout; reconstruct it at concrete values "
            "with rc.solve(params={...}).field(...)."
        )


class _RcwaProblem:
    """An RCWA problem inferred from a jNO constraint list / FEM problem. Holds the inferred
    :class:`RcwaSpec` and builds the fmmax engine on :meth:`solve`."""

    def __init__(self, spec, orders, formulation="JONES_DIRECT_FOURIER", resample=None, rpe=None):
        self.spec = spec
        self.orders = orders
        self.formulation = formulation
        self._resample = resample  # params -> ([(thickness, eps_grid), ...], wavelength)  (jax)
        self._rpe = rpe or {}  # {name: parameter-expr} for the trainable parameters in the graph

    def __repr__(self):
        return f"_RcwaProblem(orders={self.orders}, params={sorted(self._rpe)}, {self.spec!r})"

    def _engine(self):
        if getattr(self, "_eng", None) is None:  # build ONCE (warmed eagerly in _node) so the expansion is trace-free
            self._eng = Rcwa(
                self.spec.layers,
                period=self.spec.period,
                orders=self.orders,
                wavelength=self.spec.wavelength,
                k_in=self.spec.k_in,
                formulation=self.formulation,
                assume_periodic=True,
            )
        return self._eng

    def _solve_at(self, params):
        """Concrete solve at explicit parameter values (JAX) -- re-derives eps, wavelength, k_in AND (for an
        internal-source problem) the dipole current."""
        eng = self._engine()
        if not params or self._resample is None:
            return eng.solve(source=self.spec.source)
        layers, wl, kin, source = self._resample(params)
        return eng.solve(layers=layers, wavelength=wl, k_in=kin, source=source)

    def solve(self, params=None, wavelength=None, k_in=None):
        """Solve.

        The jNO way (**no arguments**): if the constraint list carries ``jno.np.parameter`` coefficients
        (ε, wavelength, ...), ``rc.solve()`` returns a solution whose ``.efficiency(...)`` / ``.order(...)``
        are **trace nodes over those parameters** -- differentiable through crux with no solve args, so
        the exact same parameterised constraint list is differentiable whether you hand it to ``jno.fem``
        or ``jno.rcwa``. A parameter-free problem solves eagerly and returns concrete readouts.

        Escape hatches for a *forward sweep* (not the traced/optimised path): ``params={name: value}`` for
        a concrete solve at explicit values, or ``wavelength`` / ``k_in`` to override those directly."""
        if params is None and wavelength is None and k_in is None:
            if self._rpe and self._resample is not None:
                return _ParametricSol(self, self._rpe)  # parametric -> trace node (crux-differentiable)
            return self._engine().solve(source=self.spec.source)  # parameter-free -> eager concrete
        eng = self._engine()
        if params is None:
            return eng.solve(wavelength=wavelength, k_in=k_in, source=self.spec.source)
        if self._resample is None:
            raise RcwaError(
                "differentiable solve(params=...) needs the analytic re-sampling path, which is only built "
                "for an analytic permittivity (no per-node field). This problem uses a nodal-field "
                "permittivity; differentiable nodal re-sampling is not implemented yet."
            )
        layers, wl, kin, source = self._resample(params)
        return eng.solve(
            layers=layers,
            wavelength=wavelength if wavelength is not None else wl,
            k_in=k_in if k_in is not None else kin,
            source=source,
        )


def _build_emitter_problem(
    femobj,
    domain,
    period,
    axes,
    ambients,
    coeff_node,
    cparams,
    coeff_layers,
    zmids,
    zspans,
    z_range,
    grid,
    nz,
    wavelength,
    orders,
    formulation,
    vsrc,
    is_aniso,
    is_pml,
):
    """Build an internal-source (dipole / Gaussian) emission problem: localize the traced ``- f·v`` /
    ``- inner(J, v)`` forcing, split its layer, and route it to fmmax's ``amplitudes_for_source``.

    The Floquet-periodic supercell + absorbing z-ambients behaves as an emitter in a layered environment.
    Scope: scalar Helmholtz layers (no tensor ε̂ / PML yet); the amplitude/orientation (and the design ε) are
    differentiable, the source *location* is static. ``k_in`` is nudged off the singular Γ-point."""
    bottom, top = ambients
    src_node, is_vector = vsrc
    if is_aniso or is_pml:
        raise RcwaError("internal-source RCWA supports the scalar Helmholtz form only (no tensor ε̂ / PML yet).")
    try:  # a boundary plane-wave incidence AND an internal source is an ambiguous double excitation
        _source_kin(femobj, domain, cparams)
        raise RcwaError("RCWA found both an internal source and a boundary plane-wave incidence; author one excitation.")
    except RcwaError as e:
        if "internal source and a boundary" in str(e):
            raise

    # order top-first: superstrate (the "up" ambient) = the max-z face; substrate ("down") = min-z
    o = list(range(len(coeff_layers)))
    if z_range[0] < z_range[1]:  # detect_layers is z-ascending -> reverse so layers[0] is the top ambient
        o = o[::-1]
    cl = [coeff_layers[i] for i in o]
    zm = [zmids[i] for i in o]
    zsp = [zspans[i] for i in o]

    super_coeff = float(np.real(cl[0][1]).mean())  # top ambient (vacuum) ⇒ coeff = k0²
    if wavelength is not None:
        k0 = 2 * np.pi / float(wavelength)
    elif super_coeff <= 0:
        raise RcwaError("cannot infer wavelength: the superstrate permittivity is non-positive; pass wavelength=.")
    else:
        k0 = float(np.sqrt(super_coeff))
    layers = [(t, np.asarray(e) / (k0 * k0)) for t, e in cl]  # scalar relative permittivity

    loc = _localize_source(src_node, is_vector, period, z_range, grid, nz, cparams)
    si = tu = tl = None
    for i in range(1, len(zsp) - 1):  # the source sits in a finite (non-ambient) layer
        lo, hi = zsp[i]
        if lo - 1e-9 <= loc["z0"] <= hi + 1e-9:
            si, tu, tl = i, float(hi - loc["z0"]), float(loc["z0"] - lo)
            break
    if si is None:
        raise RcwaError(
            f"the internal source at z={loc['z0']:.3f} is not inside a finite (non-ambient) layer; place it "
            "within a slab whose permittivity differs from the ambients (a layer contrast defines the slab)."
        )

    def comp_at(params):  # the source field's components at the centroid -> the dipole current (jx,jy,jz)
        val = jnp.reshape(_eval_coeff_points(src_node, np.array([[loc["x0"], loc["y0"], loc["z0"]]]), params), (-1,))
        if is_vector:
            return (val[0], val[1], val[2])
        return (val[0], jnp.asarray(0.0 + 0j), jnp.asarray(0.0 + 0j))  # scalar monopole -> in-plane (x) dipole

    def make_source(params):
        return dict(
            x0=loc["x0"],
            y0=loc["y0"],
            layer=si,
            t_upper=tu,
            t_lower=tl,
            kind=loc["kind"],
            fwhm=loc["fwhm"],
            orient=comp_at(params),
            amp=1.0,
        )

    kin = (1e-3, 1e-3)  # nudge off the singular Γ-point (exactly-normal k_in is degenerate for an internal source)

    def resample(params):
        k0p = jnp.sqrt(jnp.mean(jnp.real(_eval_coeff_points(coeff_node, _cell_grid_at_z(period, grid, zm[0]), params))))
        out = []
        for (thick, _), zmid in zip(layers, zm):
            cvals = jnp.reshape(_eval_coeff_points(coeff_node, _cell_grid_at_z(period, grid, zmid), params), (grid, grid))
            out.append((thick, jnp.real(cvals) / (k0p * k0p)))
        return out, 2 * jnp.pi / k0p, kin, make_source(params)

    from jno.utils.solver.parametric_helpers import _collect_runtime_parameter_exprs

    rpe = {}
    _collect_runtime_parameter_exprs(coeff_node, rpe)  # eps / wavelength parameters
    _collect_runtime_parameter_exprs(src_node, rpe)  # a trainable source amplitude / orientation
    spec = RcwaSpec(
        period=period,
        layers=layers,
        wavelength=2 * np.pi / k0,
        k_in=kin,
        source_face=top,
        ambient_faces=(bottom, top),
        periodic_axes=axes,
        source=make_source(cparams),
    )
    return _RcwaProblem(spec, orders=orders, formulation=formulation, resample=resample, rpe=rpe)


def rcwa(problem, *, orders, wavelength=None, grid=64, nz=64, slices=None, params=None, formulation="JONES_DIRECT_FOURIER"):
    """Infer and build an RCWA problem from a jNO constraint list (or built ``FEM``).

    Everything is read out of the traced problem: **periodicity + period** (Floquet ties — absent ⇒
    raise), the **super/substrate ambients**, the **permittivity** (the ``K0**2*eps`` coefficient
    recovered from the scalar Helmholtz volume term, sampled along z; a tensor ``ε̂`` from
    ``inner(ε̂ @ u, v)``), the **wavelength** (``k0`` from the vacuum superstrate, unless ``wavelength``
    is given), and the **incident wave** (illuminated face + transverse angle ``k_in``, from the
    assembled forcing). An **in-plane PML** (a complex coordinate stretch in the stiffness coefficients)
    is detected and honoured as a diagonal Maxwell ``ε̂``/``μ̂`` — the supercell then behaves as an
    isolated scatterer. Scope: uniaxial (diagonal) in-plane stretch only, scalar Helmholtz, forward
    solve (a design parameter *through* a PML is not yet differentiable — it raises).

    An **internal source** (a ``- f·v`` / ``- inner(J, v)`` volume forcing) switches the solve to an
    emission problem: the source is localized (point vs Gaussian; dipole orientation from ``J``), the stack
    is split at the source plane, and ``sol.power("up"|"down"|"total")`` / ``sol.extraction("up"|"down")``
    give the radiated power and directionality. The amplitude/orientation and design ε are differentiable;
    the source *location* is static, ``k_in`` is nudged off the singular Γ-point, and the source must sit in
    a finite (contrast-defined) layer. Scalar Helmholtz only for now; a boundary incidence + internal source
    together raise.

    Parameters
    ----------
    problem:
        The same constraint list you would pass to :func:`jno.fem`, or an already-built ``FEM``.
    orders:
        Fourier truncation for the modal solve.
    wavelength:
        Free-space wavelength (same length unit as the geometry). Optional — inferred from the
        superstrate (assumed vacuum) when omitted. Pass it to override or if no ambient is vacuum.
    grid, nz:
        Transverse resolution and number of z-samples used to detect the layer stack.
    slices:
        Staircase a continuously-varying permittivity into this many layers instead of raising.

    Returns
    -------
    _RcwaProblem
        Holds the inferred :class:`RcwaSpec` (``.spec``) and a :meth:`~_RcwaProblem.solve`.
    """
    femobj = _as_fem(problem)
    domain = femobj.domain
    axes, period = _periodic_period(femobj, domain)
    bottom, top = _z_ambient_faces(domain)
    pts = np.asarray(domain.points)
    z_range = (float(pts[:, 2].min()), float(pts[:, 2].max()))

    # permittivity coefficient K0^2 * eps, recovered from the volume weak form (no assembly, no eps arg)
    coeff_node = _extract_permittivity_coeff(_volume_constraint(femobj))
    # construction-time parameter values: the trainable parameters' current values, overridden by any
    # `params=` the caller supplied -- so the eager inference (structure, source, k0) uses valid values
    # (e.g. a K0 parameter that must be non-zero for the source to exist).
    cparams = _param_values(femobj, coeff_node)
    cparams.update({k: jnp.asarray(v) for k, v in (params or {}).items()})
    is_aniso = _coeff_is_tensor(coeff_node, period, z_range, cparams)  # 3×3 ε̂ vs scalar ε
    stretch = _extract_pml_stretch(_volume_constraint(femobj))  # {axis: Λ node} for an in-plane PML, else {}
    is_pml = bool(stretch)
    if is_pml and is_aniso:
        raise RcwaError("RCWA sees both an in-plane PML stretch and a tensor ε̂ mass term; combine them by hand.")
    if _has_nodal_field(coeff_node):  # per-node design field -> interpolate off the mesh
        coeff_nodes = _eval_expr_nodes(coeff_node, domain)
        if np.iscomplexobj(coeff_nodes) and np.max(np.abs(coeff_nodes.imag)) < 1e-9:
            coeff_nodes = coeff_nodes.real.astype(complex)
        C, zs = _sample_grid(domain, coeff_nodes, grid, nz, period, z_range)
    else:  # analytic permittivity -> sample the grid exactly
        C, zs = _sample_grid_direct(coeff_node, grid, nz, period, z_range, cparams)
    coeff_layers = detect_layers(C, zs, slices=slices)
    zmids = list(detect_layers.last_zmid)  # representative z of each layer (both paths) -> re-sampling
    zspans = list(detect_layers.last_zspan)  # (z_lo, z_hi) per layer -> place an internal source in its layer

    # ---- internal source (dipole / Gaussian emitter): a `- f·v` / `- inner(J, v)` volume forcing ----
    vsrc = _extract_volume_source(_volume_constraint(femobj))
    if vsrc is not None:
        return _build_emitter_problem(
            femobj,
            domain,
            period,
            axes,
            (bottom, top),
            coeff_node,
            cparams,
            coeff_layers,
            zmids,
            zspans,
            z_range,
            grid,
            nz,
            wavelength,
            orders,
            formulation,
            vsrc,
            is_aniso,
            is_pml,
        )

    face_z, k_in = _source_kin(femobj, domain, cparams)
    source_at_bottom = abs(face_z - _region_centroid(domain, bottom)[2]) < abs(face_z - _region_centroid(domain, top)[2])
    source_face = bottom if source_at_bottom else top

    # orient so the incident (source-side) ambient is layers[0] = superstrate
    if not source_at_bottom:
        coeff_layers = list(reversed(coeff_layers))
        zmids = list(reversed(zmids))
    # k0 from the superstrate (vacuum ⇒ coeff = k0^2), unless wavelength is given. With a PML the superstrate
    # is a vacuum framed by the absorber, so its transverse MEAN is stretched; read k0 at the physical-core
    # centre instead (there Λ=1 and ε=1, so the mass coefficient is exactly k0²).
    if is_pml:
        center = np.array([[period[0] * 0.5, period[1] * 0.5, zmids[0]]])
        super_coeff = float(np.real(_eval_coeff_points(coeff_node, center, cparams)).reshape(-1)[0])
    else:
        super_coeff = float(np.real(coeff_layers[0][1]).mean())
    if wavelength is not None:
        k0 = 2 * np.pi / float(wavelength)
    elif super_coeff <= 0:
        raise RcwaError(
            "cannot infer wavelength: the superstrate permittivity coefficient is non-positive; pass wavelength=."
        )
    else:
        k0 = float(np.sqrt(super_coeff))
    k0sq = k0 * k0
    if is_pml:  # in-plane uniaxial PML: general-anisotropic layer (ε̂ AND μ̂), 10 grids per layer
        layers = [
            (t, tuple(np.asarray(c) for c in _pml_layer_components(coeff_node, stretch, period, grid, zm, cparams, k0sq)))
            for (t, _), zm in zip(coeff_layers, zmids)
        ]
    elif is_aniso:  # relative-permittivity TENSOR per layer: (ε_xx, ε_xy, ε_yx, ε_yy, ε_zz)
        layers = [
            (t, tuple(np.asarray(c) for c in _tensor_components(coeff_node, period, grid, zm, cparams, k0sq)))
            for (t, _), zm in zip(coeff_layers, zmids)
        ]
    else:
        layers = [(t, np.asarray(e) / k0sq) for t, e in coeff_layers]  # relative permittivity (scalar)

    # differentiable re-sampling: re-evaluate the permittivity coefficient at each layer's representative z
    # from a parameter dict, so a design (jno.np.parameter) flows through the solve. Re-deriving the WHOLE
    # coefficient K0^2*eps handles an EPS/shape parameter (each layer's eps) AND a WAVELENGTH parameter
    # (k0 = sqrt(coeff in the vacuum superstrate)) in one shot.
    #
    # A per-node (free-form / topology-optimisation) density field is interpolated to the RCWA grid by a
    # precomputed node->grid BARYCENTRIC interpolation: the (node ids, weights) are fixed geometry, so
    # ``sum_i w_i * rho_nodes[id_i]`` is a smooth, differentiable interpolation of the nodal density onto the
    # RCWA grid (gradient flows to the 4 containing-tet nodes, weighted -- not just the single nearest node).
    thicks = [t for t, _ in coeff_layers]
    nodal_names = {
        mc.model._parameter_name for mc in _model_calls(coeff_node) if getattr(mc.model, "_fem_field", None) == "node"
    }
    bary = None
    if nodal_names:
        nc = np.asarray(domain.points)
        bary = [_bary_weights(nc, _cell_grid_at_z(period, grid, zm)) for zm in zmids]  # per layer: (ids, weights)

    # the illuminated boundary term's forcing -> lets an incidence-ANGLE parameter in the source flow
    # (its transverse phase gradient IS k_in); None when the source has no angle to read.
    src_coeff = next(
        (sc for c in getattr(femobj, "_constraints", []) if (sc := _extract_source_coeff(c)) is not None), None
    )
    src_z = float(_region_centroid(domain, source_face)[2])
    if src_coeff is not None:  # the source field's phase is the robust k_in (a b-based read is edge-sign-
        k_in = tuple(float(x) for x in _kin_from_source(src_coeff, period, src_z, cparams))  # corrupted for N1E)

    def resample(params):
        def at(li):  # parameter values for layer li: nodal fields barycentrically interpolated, scalars as-is
            def gather(v):
                ids, w = bary[li]
                return jnp.sum(jnp.asarray(w) * jnp.asarray(v).reshape(-1)[jnp.asarray(ids)], axis=1)

            return {k: (gather(v) if k in nodal_names else jnp.asarray(v)) for k, v in params.items()}

        if is_aniso:  # k0 from the superstrate's xx (isotropic vacuum ⇒ xx = k0²); tensor components per layer
            sup_xx = jnp.reshape(
                _eval_coeff_points(coeff_node, _cell_grid_at_z(period, grid, zmids[0]), at(0)), (grid, grid, 3, 3)
            )
            k0 = jnp.sqrt(jnp.mean(jnp.real(sup_xx[..., 0, 0])))
            out = [
                (thick, _tensor_components(coeff_node, period, grid, zmid, at(li), k0 * k0))
                for li, (thick, zmid) in enumerate(zip(thicks, zmids))
            ]
        else:
            sup = jnp.mean(jnp.real(_eval_coeff_points(coeff_node, _cell_grid_at_z(period, grid, zmids[0]), at(0))))
            k0 = jnp.sqrt(sup)
            out = []
            for li, (thick, zmid) in enumerate(zip(thicks, zmids)):
                cvals = jnp.reshape(
                    _eval_coeff_points(coeff_node, _cell_grid_at_z(period, grid, zmid), at(li)), (grid, grid)
                )
                out.append((thick, jnp.real(cvals) / (k0 * k0)))
        kin = _kin_from_source(src_coeff, period, src_z, params) if src_coeff is not None else jnp.asarray(k_in)
        return out, 2 * jnp.pi / k0, kin, None  # source=None: plane-wave incidence, not an internal source

    # the trainable jno.np.parameter(s) the permittivity depends on -> solve() returns trace nodes over
    # them (crux threads the values), so the parameterised constraint list is differentiable with no solve
    # args -- the same way jno.fem would differentiate it. (Collected from the bare coefficient tree; the
    # ScalarView-wrapped raw constraints aren't traversed by the collector.)
    from jno.utils.solver.parametric_helpers import _collect_runtime_parameter_exprs

    rpe = {}
    _collect_runtime_parameter_exprs(coeff_node, rpe)  # eps / wavelength parameters
    if src_coeff is not None:
        _collect_runtime_parameter_exprs(src_coeff, rpe)  # an incidence-angle parameter in the source
    if is_pml and rpe:  # the eager PML layers solve fine; a design parameter through a PML re-sample is future work
        raise RcwaError(
            "differentiable PML is not yet supported: a jno.np.parameter flows through an in-plane PML "
            f"problem (parameters {sorted(rpe)}). The forward PML solve works; drop the parameter to use it."
        )

    spec = RcwaSpec(
        period=period,
        layers=layers,
        wavelength=2 * np.pi / k0,
        k_in=k_in,
        source_face=source_face,
        ambient_faces=(bottom, top),
        periodic_axes=axes,
    )
    return _RcwaProblem(spec, orders=orders, formulation=formulation, resample=resample, rpe=rpe)
