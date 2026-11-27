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
      rc.spec                              # the inferred RcwaSpec (no RCWA backend pulled in)

  Inferred from the problem: **periodicity + period** (from the Floquet ties — absent ⇒ raise, never
  assumed), the **super/substrate ambients** (the z-normal radiation faces), the **permittivity**
  (the ``K0**2*eps`` coefficient recovered from the scalar Helmholtz volume term, sampled along z and
  grouped by :func:`detect_layers`), the **wavelength** (``k0`` from the vacuum ambient, the one with the smaller coefficient, or pass
  ``wavelength=`` to override), and the **incident wave** (illuminated face + transverse angle
  ``k_in``, read from the assembled forcing).

* **Explicit layers (the backend).** :class:`Rcwa` takes a hand-built ``[(thickness, eps), ...]`` stack;
  this is what the front door constructs internally.

Built on the eig-free, state-space RCWA backend in :mod:`jno._rcwa` (self-owned, no fmmax import at
runtime): each layer is solved by a first-order state transfer ``d/dz ψ = i M_L ψ`` (``expm``, never a
dense eigendecomposition) and stacked with a chunked R-matrix/impedance recursion. The previous fmmax
backend remains only as a *test-only oracle* (the cross-check suite and committed golden references).
It **never fails silently**: every inference (periodicity, layer invariance, order/grid Nyquist,
wavelength, energy balance, a recognisable source) is validated and raises :class:`RcwaError` with a
concrete fix. Two flux conventions are baked in: the Poynting flux is summed *with sign* (an
``abs()`` sum over-counts and yields ``T+R>1``), and ``JONES_DIRECT_FOURIER`` is the default
factorization (the naive rule converges poorly for high-contrast dielectrics such as a-Si).

Beyond a scalar ε, the front door also infers a **tensor permittivity** ``ε̂`` (from ``inner(ε̂ @ u, v)``)
and an **in-plane PML** (a complex coordinate stretch ``S = 1 + iσ/k`` written into the stiffness
coefficients of the scalar Helmholtz term). A uniaxial PML is exactly a diagonal Maxwell ``ε̂`` and ``μ̂``
(``ε̂ = ε·Λ``, ``μ̂ = Λ``, ``Λ = diag(SᵧS_z/Sₓ, SₓS_z/Sᵧ, SₓSᵧ/S_z)``), so the traced stretch is honoured
exactly and solved via the backend's anisotropic layer kind — turning a periodic supercell into an
isolated scatterer (the absorbing frame decouples the walls). Both the anisotropic 5/10-component
layers and the internal-source layers are wired in the backend; a patterned SCALAR-grid ambient (a
non-tensor, in-plane-varying superstrate) remains a loud stage-2 exception (task T30).

An **internal source** — a dipole / Gaussian emitter — is authored the same way you would drive `jno.fem`
or a PINN: a forcing term in the residual, ``- f·v`` (scalar) or ``- inner(J, v)`` (vector current). The
front door detects the trial-free/test-present volume summand, localizes it (centroid → point vs Gaussian
width; which z-layer), splits the stack at the source plane, and drives the backend's Maxwell-jump
source solve. Then ``sol.power("up"|"down")`` and ``sol.extraction(...)`` give the emitted power and
directionality (LED / Purcell physics) instead of plane-wave ``T``/``R``.

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
   modal dipole sources in periodic media — the internal-source (jump-condition) formulation.
 * The fmmax test-only oracle (the reference for the committed golden parity): https://github.com/facebookresearch/fmmax
"""

import functools
import warnings
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from jno._rcwa import (
    Formulation,
    LatticeVectors,
    StateStack,
    generate_expansion,
)
from jno._rcwa.stack import _Result  # the backend's request/result container for the seam readouts
from jno.litho import Threshold  # default resist for the .develop() / .printed() lithography readout
from jno.utils.profiling import annotate as _annot  # stage annotator for rc.solve(profile=True)


def _concrete(x):
    """True when ``x`` is a real (non-traced) value, so a never-silent guard may inspect it. Under
    jit/grad the value is a tracer -- the guard steps aside (validation ran on the eager forward pass)."""
    return not isinstance(x, jax.core.Tracer)


# Relative spread below which a permittivity grid counts as uniform (see ``Rcwa._prepare_stack``).
# Tight on purpose: the collapse is exact only for a genuinely constant layer, never a "nearly flat" one.
_UNIFORM_RTOL = 1e-12


class RcwaError(ValueError):
    """Raised for any ill-posed RCWA problem — always loud, never a silent wrong answer."""


class RcwaWarning(UserWarning):
    """Issued from inside ``jax.jit``/``grad``/``vmap`` where :class:`RcwaError` would be raised eagerly.

    A compiled program cannot raise a Python exception on a computed value, and a design loop should see the
    problem rather than die in an opaque runtime error; the eager path still raises."""


#: Largest accepted T + R - 1 (the house energy guard).
_ENERGY_TOL = 5e-3


def _efficiency_problem(T, R, orders, period, wl):
    """Host-side check of plane-wave efficiencies: the message for a non-finite or energy-violating result,
    or ``None``. ``T``/``R`` may be batched (a vmapped solve): one bad member is enough."""
    t, r = np.asarray(T, float), np.asarray(R, float)
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(r))):
        return (
            "non-finite efficiency (NaN/inf): likely a Rayleigh/Wood anomaly where a diffraction "
            "order is exactly grazing (the wavelength coincides with a period resonance, e.g. "
            f"wavelength == period). Nudge the wavelength or period. (period={period}, wl={wl})"
        )
    total = float(np.max(t + r))
    if total > 1.0 + _ENERGY_TOL:
        return (
            f"energy not conserved: T+R={total:.4f} > 1 at orders={orders}. "
            "The state-space solve is not converged: raise orders= for more Fourier orders, or "
            "-- for a thick high-contrast layer -- thin the sub-slab chunking (the T+R>1 cliff "
            "is the un-chunked impedance recursion; auto chunking is fixed at build, so rebuild "
            "the Rcwa if chunking is the culprit), or the structure is unphysical."
        )
    return None


def _warn_on_efficiency_problem(T, R, *, orders, period, wl):
    """The traced counterpart of the eager guard: a host callback that warns (see :class:`RcwaWarning`)."""
    msg = _efficiency_problem(T, R, orders, period, wl)
    if msg is not None:
        warnings.warn(msg, RcwaWarning, stacklevel=2)


# =====================================================================================
# Layer auto-detection: group z-invariant slabs of a sampled permittivity into RCWA layers.
# =====================================================================================
def detect_layers(E, z, tol=1e-3, slices=None, locate=None):
    """Group a z-sampled permittivity into RCWA layers.

    Args:
        E: ``(Nz, Ng, Ng)`` permittivity sampled on the unit-cell grid at each height ``z``.
        z: ``(Nz,)`` ascending heights.
        tol: In-plane change below which two adjacent z-slices are the same material.
        slices: If given, staircase a continuously-varying ``eps`` into this many layers instead of
            raising.
        locate: ``locate(z_lo, z_hi) -> z`` places the interface between two neighbouring layers, given
            heights inside each (their middle samples). Without it the interface is put halfway between
            the last sample of one layer and the first of the next, so a thickness is only known to one
            sample spacing. The front door passes a bisection on the permittivity itself, which finds a
            sharp interface exactly and a mesh field's interface where its interpolant crosses halfway
            between the layers. A layer's thickness is the distance between the interfaces that bound
            it -- not between its first and last samples, which is short by up to one spacing (a 0.2
            slab sampled every 1/63 came out 11/63 = 0.175 thick, and its reflectance 0.34 against
            Airy's 0.62).

    Returns:
        list[tuple[float, np.ndarray]]: ``[(thickness, eps_xy), ...]`` super→substrate; the two ambients
            have thickness ``inf``.

    Raises:
        RcwaError: If ``eps`` varies continuously in z and ``slices`` is None (RCWA needs invariant
            slabs).
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

    # interface heights: slab i spans [iface[i], iface[i + 1]]; the outer two are the cell's ends
    iface = [float(z[0])]
    for (a0, b0), (a, b) in zip(slabs[:-1], slabs[1:]):
        if locate is not None:
            iface.append(float(locate(float(z[(a0 + b0) // 2]), float(z[(a + b) // 2]))))
        else:
            iface.append(0.5 * float(z[a - 1] + z[a]))
    iface.append(float(z[-1]))

    layers, report, zmids, zspans = [], [], [], []
    for i, (a, b) in enumerate(slabs):
        mid = (a + b) // 2
        var = float(np.max(np.abs(E[a:b] - E[mid])))
        if var > tol and slices is None:
            raise RcwaError(
                f"slab z=[{z[a]:.3f},{z[min(b, Nz - 1)]:.3f}] is not z-invariant (var={var:.2g}); pass slices=."
            )
        thick = np.inf if (i == 0 or i == len(slabs) - 1) else iface[i + 1] - iface[i]
        eps_xy = E[mid]
        kind = "uniform" if np.ptp(eps_xy) < tol else "patterned"
        layers.append((thick, eps_xy))
        zmids.append(float(z[mid]))
        zspans.append((iface[i], iface[i + 1]))  # (z_lo, z_hi) -> place an internal source
        report.append(
            f"  layer {i}: z=[{iface[i]:.4f},{iface[i + 1]:.4f}] {kind} eps~[{eps_xy.min():.2f},{eps_xy.max():.2f}]"
        )
    detect_layers.last_report = "detected layers:\n" + "\n".join(report)
    detect_layers.last_zmid = zmids  # representative z of each layer -- lets a param sweep re-sample eps
    detect_layers.last_zspan = zspans  # (z_lo, z_hi) per layer -- lets an internal source split its layer
    return layers


# =====================================================================================
# The forward engine (explicit layers) — the state-space backend the front door constructs.
# =====================================================================================
def _as_jones(pol):
    """A transverse Jones vector ``(p_x, p_y)`` of unit norm from ``"x"``, ``"y"`` or a 2-sequence (complex
    entries give elliptical polarization). Raises on anything else, and on a zero vector."""
    if isinstance(pol, str):
        if pol == "x":
            return (1.0 + 0j, 0j)
        if pol == "y":
            return (0j, 1.0 + 0j)
        raise RcwaError(f"polarization must be 'x', 'y' or a Jones vector (px, py), got {pol!r}")
    try:
        px, py = (complex(c) for c in pol)
    except (TypeError, ValueError):
        raise RcwaError(f"polarization must be 'x', 'y' or a Jones vector (px, py), got {pol!r}") from None
    n = float(np.sqrt(abs(px) ** 2 + abs(py) ** 2))
    if n == 0.0:
        raise RcwaError("polarization Jones vector is zero; give (px, py) with |px|^2 + |py|^2 > 0.")
    return (px / n, py / n)


def _incident_pol(pol):
    """The backend's incident plane-wave state for the Jones pair ``pol`` is built *inside* the
    jitted core (``u_inc = (-E_y, E_x)`` at order (0,0), ``v_inc = Y_sup u_inc``) -- the same uniform
    real-space wave the older engine decomposed via ``amplitudes_for_fields``. What the readouts need
    from ``pol`` is only the pair itself, re-solved through the compiled core when a readout asks for a
    different input polarization (``jones`` / the vector-aerial pupil)."""
    return _as_jones(pol)


class _Sol:
    def __init__(self, eng, res, expansion, nt, Pin, wavelength, thick=None, period=None, pol=(1.0, 0.0)):
        self._eng = eng  # the Rcwa engine: re-solve through its compiled core for an alternate input pol
        self._s = res  # the backend _Result (T/R/pin/state vectors) of THIS solve
        self._ex = expansion
        self._nt, self._Pin, self._wl = nt, Pin, wavelength
        self._thick, self._period = thick, period
        self._pol = pol  # the incident Jones vector every readout refers to

    def _e_of(self, state):
        """Per-order transverse E-field amplitudes ``(ex, ey)`` of a tangential state ``u = (-E_y, E_x)``."""
        return jnp.reshape(state[self._nt :], (-1,)), jnp.reshape(-state[: self._nt], (-1,))

    def efficiency(self, kind):
        """Total transmitted (``"T"``) or reflected (``"R"``) power fraction over propagating orders.

        Returns a JAX scalar -- differentiable in the design (permittivity) so ``jax.grad`` of a
        transmission/reflection objective flows through the state-space solve."""
        if kind == "T":
            return self._s.T
        if kind == "R":
            return self._s.R
        raise RcwaError(f"efficiency(kind): kind must be 'T' or 'R', got {kind!r}")

    def order(self, m, n):
        """Diffraction efficiency into transmitted order ``(m, n)`` (raises if outside the truncation).
        Returns a differentiable JAX scalar.

        An evanescent order (the substrate-side ``Re(kz) ~ 0``) reports EXACTLY 0 -- the fmmax-era
        ``is_propagating`` convention -- instead of its ~1e-17 Poynting-noise readout; the gate uses
        a tiny ``Re(kz)`` threshold (never exact 0) so fp32/fp64 rounding near the Rayleigh cutoff
        cannot leak evanescent flux. Lossy substrates keep their genuinely-carrying evanescent-in-air
        orders (their ``Re(kz) != 0``), and the ``T``/``R`` efficiency aggregates are unaffected."""
        bc = np.asarray(self._ex.basis_coefficients)  # static (truncation geometry) -> plain numpy index
        hit = np.where((bc[:, 0] == m) & (bc[:, 1] == n))[0]
        if len(hit) == 0:
            raise RcwaError(f"order ({m},{n}) is outside the truncation ({self._nt} terms); raise orders=.")
        i = int(hit[0])
        # Signed per-order z-Poynting of the transmitted state, summed over the two tangential
        # components (jNO's flux convention), normalized by the incident power. For a propagating
        # order both tangential components carry +z energy, so |sum over components| == sum of
        # |component| -- equal to the old per-component-abs form; only pathological (evanescent /
        # counter-propagating) orders differ, where the signed-sum form is the physical one.
        flux = self._s.flux_trn[i]
        eng = self._eng
        grids = getattr(eng, "_prepared_grids", None)
        if grids is not None:
            # Per-order substrate kz, on the same side medium the flux was computed against (the
            # analytic split scalar from the plane-wave seam: vacuum beyond a folded PML frame).
            _, _, eps_sub = eng._backend_materials(tuple(grids))
            kin = jnp.asarray(eng._prepared_kin)
            k0 = 2 * jnp.pi / jnp.asarray(self._wl, kin.dtype)
            gx, gy = bc[i]
            kpar2 = (kin[0] + 2 * jnp.pi * gx / self._period[0]) ** 2 + (kin[1] + 2 * jnp.pi * gy / self._period[1]) ** 2
            kz_re = jnp.real(jnp.sqrt(eps_sub * k0**2 - kpar2))  # Im(kz) >= 0 branch (uniform_kz)
            thr = (1e-10 if jnp.dtype(jnp.real(flux).dtype).itemsize >= 8 else 1e-6) * k0
            flux = jnp.where(kz_re < thr, jnp.zeros_like(flux), flux)
        return jnp.abs(flux) / self._Pin

    def _input_pol(self, pol):
        """The normalized Jones pair for an alternate input polarization (``"x"`` or ``"y"``) -- re-solved
        through the compiled core when a readout needs the response to a different input (``jones`` /
        vector-aerial pupil). The backend builds the incident state itself: the 0th-order ``(-E_y, E_x)``
        of the requested polarization at ``k_in``, i.e. the field-space decomposition of the uniform
        plane wave (jax-native, traces under grad/jit)."""
        return _as_jones(pol)

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

        def out0(pol):  # the 0th-order (E_x, E_y) of the output field for this input plane wave
            ex, ey = self._spectrum(kind, fwd=self._input_pol(pol))
            return ex[0], ey[0]  # (0,0) is expansion index 0

        exx, eyx = out0("x")  # input x-pol -> output (E_x, E_y)
        exy, eyy = out0("y")  # input y-pol -> output (E_x, E_y)
        return jnp.array([[exx, exy], [eyx, eyy]])  # J[out q, in p]

    def _nonfinite_field_message(self):
        """Why the interior amplitudes are NaN/inf: name the order(s) whose z-wavevector vanishes in a layer
        (a grazing order, kz = 0, makes the interior propagation 1/kz blow up while T and R stay finite)."""
        bc = np.asarray(self._ex.basis_coefficients)
        grazing = []
        for i, layer in enumerate(self._layers):
            ev = np.abs(np.asarray(layer.eigenvalues))  # per order and polarization: ~ |kz|
            near = np.where(ev < 1e-4 * max(float(ev.max()), 1e-300))[0]
            grazing += [(i, tuple(int(c) for c in bc[j % self._nt])) for j in near]
        what = (
            "; ".join(f"order {o} in layer {i}" for i, o in dict.fromkeys(grazing))
            if grazing
            else "no single grazing order was identified"
        )
        return (
            "the interior field is non-finite (NaN/inf) although T and R are finite: an order sits at grazing "
            f"incidence (kz = 0), the Rayleigh/Wood condition ({what}; period={self._period}, wl={self._wl}). "
            "Shift the wavelength, the incidence angle or the period slightly (e.g. a period of 2.1 instead of "
            "2.0 wavelengths), or lower orders= so the truncation drops that order."
        )

    def _field_volume(self, nx=80, density=40.0):
        """``(|E|^2, x, y, z)`` on the full 3-D reconstruction grid — the shared core of :meth:`field`
        and :meth:`field3d`. ``|E|^2`` has shape ``(nx, nx, nz)`` (z along the last axis); ``x``/``y``
        are 1-D axis coordinates and ``z`` the per-plane depths (1-D, endpoints shared between
        neighbouring layers, cumulative from the superstrate top surface).

        The interior reconstruction lives in the backend (``jno/_rcwa/fields.py``): the boundary
        state is propagated to every plane through the stack's own chunked/impedance machinery and
        inverse-Fourier-transformed here. Lossless anchors and the absorbed-power/z-profile tests
        pin the energy balance through ``|E|^2``.
        """
        eng = self._eng
        if self._thick is None:
            raise RcwaError("field() needs the layer thicknesses; call .solve() (which records them).")
        if eng._stack is None or eng._prepared_grids is None:
            raise RcwaError("field() needs a solved stack; call .solve() first.")
        if not (isinstance(nx, (int, np.integer)) and nx >= 2):
            raise RcwaError(f"field(): the render grid nx must be an int >= 2, got {nx!r}.")
        if not (isinstance(density, (int, float, np.floating)) and float(density) > 0):
            raise RcwaError(f"field(): density (z samples per unit thickness) must be > 0, got {density!r}.")
        from jno._rcwa import fields as _fields_mod

        thicknesses = [float(jnp.real(jnp.asarray(t))) for t in self._thick]
        znums, z = _fields_mod.layer_z_sampling(thicknesses, density)
        n_full = len(thicknesses)
        fold_top, fold_bot = getattr(eng, "_fold_amb", (False, False))
        roles: list = []
        film_count = 0
        for i in range(n_full):
            if i == 0 and not fold_top:
                roles.append("sup")
            elif i == n_full - 1 and not fold_bot:
                roles.append("sub")
            else:
                roles.append(film_count)
                film_count += 1
        stack = eng._stack
        planes = _fields_mod.field_planes(thicknesses, znums, roles, stack._chunks)
        try:
            _fields_mod.validate_grid_resolution(nx, eng._ex, "field3d")
        except ValueError as e:
            raise RcwaError(str(e)) from e
        films, sup, sub = eng._backend_materials(eng._prepared_grids)
        (ex, ey, ez, _hx, _hy, _hz) = _fields_mod.interior_field_amplitudes(
            stack,
            planes,
            films,
            eng._prepared_wl,
            eng._prepared_kin,
            self._pol,
            sup,
            sub,
        )
        inten, x, y = _fields_mod.volume_intensity(ex, ey, ez, eng._ex, eng._prepared_kin, self._period, nx)
        return inten, x, y, jnp.asarray(z)

    def field(self, y_frac=0.5, nx=80, density=40.0):
        """Reconstruct the real-space electric field on a vertical (x–z) slice at ``y = y_frac·Py``.

        Returns ``(intensity, extent, layer_z)``: ``intensity`` = ``|E|²`` of shape ``(nz, nx)`` (z
        vertical, x horizontal) for ``imshow``; ``extent`` = ``[0, Px, z_min, z_max]``; ``layer_z`` = the
        z-interfaces between layers (to annotate the patterned slab). The finite ambient
        thickness recorded at solve time is shown (semi-infinite ambients as unit-thick slabs).

        Implemented by the backend's interior field reconstruction (task T31): ``|E|²`` is a JAX
        array, differentiable in the design.

        Caveat: at OBLIQUE incidence (``k_in`` tilted) the reconstructed ``|E|²`` is ~5% off the
        accurate energy content (state-space split conditioning); efficiency/flux/absorbed-power
        readouts are unaffected and are the accurate quantities for energy work."""
        inten, x, y, z = self._field_volume(nx=nx, density=density)
        j = int(jnp.argmin(jnp.abs(y - y_frac * float(self._period[1]))))
        slab = inten[:, j, :].T  # (nz, nx): z vertical, x horizontal
        zc = np.asarray(z)
        layer_z = list(np.cumsum([float(t) for t in self._thick]))[:-1]
        return slab, [0.0, float(self._period[0]), float(zc.min()), float(zc.max())], layer_z

    def field3d(self, nx=80, density=40.0):
        """``|E|²`` on the full 3-D grid, for volume rendering / isosurfaces — the whole reconstruction
        rather than the single ``y``-slice :meth:`field` returns.

        Returns ``(intensity, (x, y, z), layer_z)`` with ``intensity`` of shape ``(nx, ny, nz)`` and
        ``x``/``y``/``z`` the 1-D coordinates of each axis. The semi-infinite ambients are rendered as
        unit-thick slabs (see :meth:`field`), so ``z`` starts at 0 at the top of the superstrate, not at
        any coordinate of the authored geometry — ``layer_z`` gives the interfaces in the same frame.

        A JAX array, differentiable in the design (the layers' permittivity grids): ``jax.grad`` of a
        functional of it (e.g. the absorbed power ``½ ω ε₀ Im ε |E|²`` heating a coupled thermal solve) flows
        back through the interior-field reconstruction and the eig-free state-space solve. The grid sizes (``nx``, the
        per-layer ``nz`` from ``density``) and the layer thicknesses are static.

        Caveat (documented + pinned in ``tests/test_rcwa_fields.py``): at OBLIQUE incidence the
        reconstructed ``|E|²`` deviates ~5% from the accurate energy content -- use the
        efficiency/absorbed-power readouts for quantitative energy work at tilted ``k_in``."""
        inten, x, y, z = self._field_volume(nx=nx, density=density)
        layer_z = list(np.cumsum([float(t) for t in self._thick]))[:-1]
        return inten, (x, y, z), layer_z

    def _spectrum(self, kind="T", fwd=None):
        """The complex diffraction spectrum for imaging: the transverse E-field Fourier coefficients
        ``(ex, ey)`` per diffraction order (transmitted ``"T"`` or reflected ``"R"``), for the corrected
        0th-order incidence (or, when ``fwd`` is an input polarization pair, re-solved for that input).
        ``(0,0)`` is expansion index 0."""
        if kind not in ("T", "R"):
            raise RcwaError(f"kind must be 'T' or 'R', got {kind!r}")
        if fwd is not None:  # an alternate input polarization: re-solve through the compiled core
            res = self._eng._resolve_pol(fwd)
        else:
            res = self._s
        state = res.u_sub if kind == "T" else res.u_ref
        return self._e_of(state)

    def aerial(self, NA, source=0.7, defocus=0.0, grid=128, kind="T", polarization=None):
        """Partially-coherent **aerial image** (wafer-plane intensity) of this mask, by **Abbe** source
        integration -- the litho imaging step on top of the rigorous RCWA mask diffraction.

        The mask's diffraction orders (this solution's spectrum) are projected through a lens of numerical
        aperture ``NA``, summed over the illumination ``source``. Everything (wavelength, period, the complex
        mask spectrum) is read from the solve; only the optics are yours.

        Args:
            NA: Numerical aperture of the projection lens (an order passes the pupil when its direction
                cosine ``|α| ≤ NA``).
            source: Illumination. A **float** ``σ`` → conventional (a filled disk of partial-coherence
                radius ``σ``); a **(σ_in, σ_out) tuple** → annular; a raw **array** of pupil weights
                (differentiable, for source-mask optimization).
            defocus: Defocus (same length unit as the geometry); applies the quadratic pupil phase.
            grid: Real-space resolution of the returned image (one period).
            kind: ``"T"`` (transmissive DUV mask, default) or ``"R"`` (reflective EUV mask).
            polarization: ``None`` (default) → **scalar** imaging of the field component along the incident
                polarization (``E_x`` for an x-polarized solve), correct at low NA. A string turns on the
                **vector high-NA** model, which rotates each order's transverse ``(E_x, E_y)`` to the 3-D
                wafer field through the Richards-Wolf/Flagello vector pupil (so the TM component loses
                contrast at large angles): ``"x"`` / ``"y"`` → linearly polarized illumination;
                ``"unpolarized"`` → the two averaged.

        Returns:
            A ``(grid, grid)`` JAX intensity array over one period -- differentiable in the mask design and
                the
            source, so ``jax.grad`` of a printed-vs-target loss drives OPC / ILT / SMO.

        Note:
            Uses the shift-invariant (Kirchhoff-like) Abbe sum on this single solve's rigorous spectrum; full
            angular rigor (re-solving the mask per source point) is a heavier option. The vector pupil follows
            Flagello, Milster & Rosenbluth, *J. Opt. Soc. Am. A* **13**, 53 (1996), with the aplanatic
            ``1/√(cosθ)`` apodization; at NA→0 it reduces to the scalar image.
        """
        wl = self._wl  # may be a tracer (a wavelength parameter) -- keep jax-native
        Px, Py = float(self._period[0]), float(self._period[1])
        bc = np.asarray(self._ex.basis_coefficients)  # (nt, 2) static (m, n)
        M = int(np.abs(bc).max())
        grid = max(int(grid), 2 * M + 2)
        ms = jnp.asarray(np.arange(-M, M + 1))
        AX, AY = jnp.meshgrid(ms * wl / Px, ms * wl / Py, indexing="ij")  # α (direction cosine) per order
        S, W = _source_grid(source, float(NA))
        c = grid // 2

        def _grid(vals):  # scatter an order-indexed spectrum onto the (2M+1)² frequency grid
            return jnp.zeros((2 * M + 1, 2 * M + 1), complex).at[bc[:, 0] + M, bc[:, 1] + M].set(vals)

        def _place(Apad):  # embed a pupil-weighted spectrum and inverse-FFT to a real-space field
            pad = jnp.zeros((grid, grid), complex).at[c - M : c + M + 1, c - M : c + M + 1].set(Apad)
            return jnp.fft.ifft2(jnp.fft.ifftshift(pad)) * (grid * grid)

        if polarization is None:  # scalar imaging: the co-polarized component of the incident Jones vector
            ex_, ey_ = self._spectrum(kind)
            px, py = self._pol
            A = _grid(np.conj(px) * ex_ + np.conj(py) * ey_)

            def one(s):  # image from a single source point s (a shift of the order frequencies)
                r2 = (AX + s[0]) ** 2 + (AY + s[1]) ** 2
                pupil = jnp.where(r2 <= NA**2, jnp.exp(-1j * jnp.pi * wl * defocus * r2), 0.0)
                return jnp.abs(_place(A * pupil)) ** 2

            imgs = jax.vmap(one)(S)  # (Ns, grid, grid)
            return jnp.sum(W[:, None, None] * imgs, axis=0) / jnp.sum(W)

        # Vector (high-NA) imaging: rotate each order's transverse (E_x, E_y) to the 3-D wafer field via the
        # Richards-Wolf/Flagello vector pupil (below). Average the incoherent images of the requested input
        # polarizations. At NA→0 the pupil is the identity and this reduces to |E_x|²+|E_y|² == the scalar image.
        pols = ("x", "y") if polarization == "unpolarized" else (polarization,)
        specs = [(_grid(ex), _grid(ey)) for ex, ey in (self._spectrum(kind, fwd=self._input_pol(p)) for p in pols)]

        def one(s):
            ax, ay = AX + s[0], AY + s[1]
            r2 = ax**2 + ay**2
            passes = r2 <= NA**2
            # sinθ / cosθ via a double-`where` safe sqrt: √ is only ever evaluated at a strictly-positive
            # argument, so its reverse-mode `1/(2√·)` never hits the 0th-order (r2=0) or grazing (r2=1)
            # singularity. rho/cth are constant in the design, but an unguarded √0 grad still poisons the sum.
            s2 = jnp.clip(r2, 0.0, 1.0)
            rho = jnp.where(s2 > 0.0, jnp.sqrt(jnp.where(s2 > 0.0, s2, 1.0)), 0.0)  # sinθ
            c2 = jnp.clip(1.0 - r2, 0.0, 1.0)
            cth = jnp.where(c2 > 0.0, jnp.sqrt(jnp.where(c2 > 0.0, c2, 1.0)), 0.0)  # cosθ
            safe = rho > 1e-9
            cf = jnp.where(safe, ax / jnp.where(safe, rho, 1.0), 1.0)  # cosφ (azimuth)
            sf = jnp.where(safe, ay / jnp.where(safe, rho, 1.0), 0.0)  # sinφ
            apod = jnp.where(  # defocus phase × aplanatic 1/√(cosθ), gated by the pupil
                passes, jnp.exp(-1j * jnp.pi * wl * defocus * r2) / jnp.sqrt(jnp.where(passes, cth, 1.0)), 0.0
            )
            m00, m01, m11 = cf**2 * cth + sf**2, cf * sf * (cth - 1.0), sf**2 * cth + cf**2
            m20, m21 = -rho * cf, -rho * sf
            tot = jnp.zeros((grid, grid))
            for Ax, Ay in specs:
                ux, uy = Ax * apod, Ay * apod
                ewx, ewy, ewz = _place(m00 * ux + m01 * uy), _place(m01 * ux + m11 * uy), _place(m20 * ux + m21 * uy)
                tot = tot + jnp.abs(ewx) ** 2 + jnp.abs(ewy) ** 2 + jnp.abs(ewz) ** 2
            return tot / len(specs)

        imgs = jax.vmap(one)(S)
        return jnp.sum(W[:, None, None] * imgs, axis=0) / jnp.sum(W)

    def expose(self, NA, source=0.7, defocus=0.0, grid=128, kind="T", polarization=None):
        """The **optical exposure** of this mask at the wafer plane -- the imaging result before any resist.

        Returns an :class:`_Exposure` (optics only: :meth:`_Exposure.intensity` = the aerial image,
        :meth:`_Exposure.spectrum` = the diffraction-order angular spectrum). Turn it into a developed
        pattern with ``exposure.develop(resist)`` for a resist model (:class:`jno.litho.Threshold`, or a
        rigorous PEB model). The imaging arguments ``NA … polarization`` are exactly :meth:`aerial`'s."""
        return _Exposure(self, (NA, source, defocus, grid, kind, polarization))

    def printed(self, NA, source=0.7, defocus=0.0, grid=128, kind="T", polarization=None, resist=None):
        """Developed **resist image** (the printed pattern) -- a one-call shortcut for
        ``expose(...).develop(resist)``. ``resist`` defaults to :class:`jno.litho.Threshold` (the fast
        constant-threshold + PEB-diffusion model). Differentiable in the mask + source, so ``jax.grad`` of a
        printed-vs-target loss drives **OPC / ILT / SMO** back through development, imaging *and* the RCWA
        mask solve. Pass a different resist model (e.g. a rigorous reaction-diffusion PEB) to swap physics
        without changing the call."""
        return self.expose(NA, source, defocus, grid, kind, polarization).develop(resist or Threshold())

    def _bulk(self, cfg, film, source_chunk=None):
        """Standing-wave **bulk image** ``|E(x, y, z)|²`` inside the resist film. Each diffraction order (a
        plane wave at direction cosine ``(α, β)``) is refracted into the film (``k_z = k0·√(n_r² − α² − β²)``,
        complex ⇒ absorption) and interferes with its substrate reflection ``r_s``, producing the vertical
        standing wave. Reuses the aerial Abbe sum with a per-order depth factor; returns ``(grid, grid, nz)``.
        Scalar (the field component along the incident polarization, ``E_x`` for an x-polarized solve);
        single-substrate-reflection model (the dominant standing wave).

        ``source_chunk`` bounds the peak memory: the Abbe sum over source points is accumulated
        ``source_chunk`` points at a time instead of materialising the whole stack (see below).
        ``None`` = all at once, the historical behaviour."""
        NA, source, defocus, grid, kind, _polarization = cfg
        wl = self._wl
        Px, Py = float(self._period[0]), float(self._period[1])
        bc = np.asarray(self._ex.basis_coefficients)
        M = int(np.abs(bc).max())
        grid = max(int(grid), 2 * M + 2)
        ex_, ey_ = self._spectrum(kind)
        px, py = self._pol  # the co-polarized component, as in the scalar aerial image
        A = (
            jnp.zeros((2 * M + 1, 2 * M + 1), complex)
            .at[bc[:, 0] + M, bc[:, 1] + M]
            .set(np.conj(px) * ex_ + np.conj(py) * ey_)
        )
        ms = jnp.asarray(np.arange(-M, M + 1))
        AX, AY = jnp.meshgrid(ms * wl / Px, ms * wl / Py, indexing="ij")
        S, W = _source_grid(source, float(NA))
        c = grid // 2
        k0 = 2.0 * jnp.pi / wl
        n_r, n_s, n_top = complex(film.n_resist), complex(film.n_substrate), complex(getattr(film, "n_top", 1.0))
        d, nz = float(film.thickness), int(getattr(film, "nz", 16))
        zs = jnp.linspace(0.0, d, nz)
        r_s, t_top = (n_r - n_s) / (n_r + n_s), 2.0 * n_top / (n_top + n_r)  # substrate reflection + top transmission

        def plane(s, z):  # one source point, one depth -> |E(x, y, z)|²
            a2 = (AX + s[0]) ** 2 + (AY + s[1]) ** 2
            kz = k0 * jnp.sqrt(n_r**2 - a2 + 0j)  # z-wavevector in the resist (complex ⇒ depth absorption)
            fz = t_top * (jnp.exp(1j * kz * z) + r_s * jnp.exp(1j * kz * (2.0 * d - z)))  # down + substrate-reflected up
            pupil = jnp.where(a2 <= NA**2, jnp.exp(-1j * jnp.pi * wl * defocus * a2) * fz, 0.0)
            pad = jnp.zeros((grid, grid), complex).at[c - M : c + M + 1, c - M : c + M + 1].set(A * pupil)
            return jnp.abs(jnp.fft.ifft2(jnp.fft.ifftshift(pad)) * (grid * grid)) ** 2

        # The Abbe sum is a WEIGHTED REDUCTION over source points, so it accumulates -- but vmapping it
        # in one shot materialises the whole ``(n_source, nz, grid, grid)`` stack first. At 192² × 40
        # depths with a σ=0.6 source that intermediate is ~5 GB and OOMs a small card, even though the
        # result is only ``(nz, grid, grid)``. Scan over CHUNKS of source points instead: peak memory
        # becomes ``O(chunk · nz · grid²)`` and ``chunk`` trades parallelism against it. ``lax.scan``
        # keeps this jit-able and differentiable (no Python unrolling).
        #
        # Padding with ZERO-weight source points keeps the scan shapes static; a zero weight adds
        # nothing, so the result is the same sum -- only the floating-point association order changes.
        ns = int(S.shape[0])
        chunk = max(1, min(int(source_chunk or ns), ns))
        pad = (-ns) % chunk
        Sp = jnp.concatenate([S, jnp.zeros((pad,) + S.shape[1:], S.dtype)]) if pad else S
        Wp = jnp.concatenate([W, jnp.zeros((pad,), W.dtype)]) if pad else W
        Sp = Sp.reshape(-1, chunk, *S.shape[1:])
        Wp = Wp.reshape(-1, chunk)
        proto = jax.eval_shape(plane, S[0], zs[0])  # shape/dtype of one plane, computed for free

        def _accum(acc, sw):
            s_c, w_c = sw
            imgs = jax.vmap(lambda s: jax.vmap(lambda z: plane(s, z))(zs))(s_c)  # (chunk, nz, g, g)
            return acc + jnp.sum(w_c[:, None, None, None] * imgs, axis=0), None

        acc0 = jnp.zeros((int(nz),) + proto.shape, proto.dtype)
        vol, _ = jax.lax.scan(_accum, acc0, (Sp, Wp))
        vol = vol / jnp.sum(W)  # (nz, grid, grid)
        return jnp.moveaxis(vol, 0, -1)  # (grid, grid, nz): x, y, depth


class _Exposure:
    """The **optical exposure** at the wafer plane (from :meth:`_Sol.expose`) -- optics only, no resist. A
    resist model (any callable ``exposure -> developed field``) is applied with :meth:`develop`; the fast
    :class:`jno.litho.Threshold` reads :meth:`intensity`, a depth-resolved resist reads :meth:`bulk` (the
    standing-wave field inside the film). Keeping the exposure resist-agnostic is what lets new resist models
    plug in without touching the imaging code."""

    def __init__(self, sol, cfg):
        self._sol, self._cfg = sol, cfg
        self.period = (float(sol._period[0]), float(sol._period[1]))

    def intensity(self):
        """The **aerial image** ``|E|²`` at the wafer -- a ``(grid, grid)`` JAX array (see :meth:`_Sol.aerial`;
        the exposure's imaging config is applied). What a constant-threshold resist develops."""
        return self._sol.aerial(*self._cfg)

    def bulk(self, film, source_chunk=None):
        """The **standing-wave bulk image** ``|E(x, y, z)|²`` inside the resist ``film`` -- the aerial spectrum
        propagated into the film so each order interferes with its substrate reflection (the vertical standing
        wave) and decays by absorption (a complex resist index). ``film`` supplies ``n_resist`` (complex for
        absorption), ``thickness``, ``n_substrate``, optional ``n_top`` (default 1) and ``nz`` (depth samples;
        default 16) -- e.g. :class:`jno.litho.Film`. Returns a ``(grid, grid, nz)`` JAX array. At ``z = 0`` with
        no substrate reflection it equals :meth:`intensity`. The depth-resolved input a rigorous 3-D resist
        develops; scalar (``E_x``), single-substrate-reflection model (full multilayer Airy is a refinement)."""
        return self._sol._bulk(self._cfg, film, source_chunk=source_chunk)

    def develop(self, resist):
        """Apply a **resist model** -- any callable ``exposure -> developed field`` (the fast
        :class:`jno.litho.Threshold`, or a rigorous PEB model). Equivalent to ``resist(self)``."""
        return resist(self)


def _source_grid(source, NA, ns=15):
    """Illumination source as (points, weights) on a pupil grid, for the Abbe sum. ``source`` is a float
    (conventional disk of coherence radius σ), a (σ_in, σ_out) tuple (annular), or a raw weight array."""
    if not np.isscalar(source) and not isinstance(source, tuple):  # a raw pupil-weight array
        w = jnp.asarray(source).reshape(-1)
        ns = int(round(float(w.size) ** 0.5))
    sp = np.linspace(-NA, NA, ns)
    SX, SY = np.meshgrid(sp, sp, indexing="ij")
    S = jnp.asarray(np.stack([SX.ravel(), SY.ravel()], 1))  # (ns², 2)
    rn = np.hypot(np.asarray(SX).ravel(), np.asarray(SY).ravel()) / NA  # coherence radius σ per point
    if np.isscalar(source):
        W = jnp.asarray((rn <= float(source)).astype(float))  # conventional disk
    elif isinstance(source, tuple):
        W = jnp.asarray(((rn >= source[0]) & (rn <= source[1])).astype(float))  # annular
    else:
        W = jnp.asarray(source).reshape(-1)  # arbitrary (differentiable for SMO)
    return S, W


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

    Args:
        layers: the stack, ``[(thickness, eps), ...]`` from top to bottom; ``eps`` is a scalar or a
            permittivity grid (2-D array), a 5-tuple ``(eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)``
            (anisotropic) or a 10-tuple adding ``(mu_xx, ...)`` (general, magnetic).
        period: the lattice periods ``(Lx, Ly)``.
        orders: Fourier truncation (approximate number of retained orders).
        wavelength: free-space wavelength; may instead be given to :meth:`solve`.
        k_in: transverse incident wavevector ``(kx, ky)``.
        formulation: the Fourier-factorization name, e.g. ``"JONES_DIRECT_FOURIER"`` (default; converges
            well for high-contrast dielectrics, where the naive ``"FFT"`` rule converges poorly). The
            state-space backend implements all eight fmmax vector schemes plus ``FFT`` and a callable
            tangent-field hook.
        assume_periodic: skip the check that each layer's grid is periodic.
        polarization: ``"x"``, ``"y"``, or a Jones vector ``(p_x, p_y)``.
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
        polarization="x",
    ):
        self.polarization = _as_jones(polarization)
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
        self.layers_spec, self.period = layers, period
        self.orders, self.wavelength, self.k_in = orders, wavelength, tuple(k_in)
        # Materials may be a scalar / grid (isotropic) or a 5-tuple (anisotropic eps)
        # / 10-tuple (eps+mu) -- all wired in the owned backend. A RAW (...,3,3)
        # tensor array is NOT a supported layer material (the front door produces
        # the component tuples); reject it loudly, never guess which 5 components
        # it stands for.
        for k, (_t, e) in enumerate(layers):
            if not isinstance(e, tuple) and jnp.ndim(e) >= 3:
                raise RcwaError(
                    f"layer {k}: a tensor-form ({tuple(jnp.shape(e))}) anisotropic layer must be "
                    "given as its 5-component (eps_xx..eps_zz) or 10-component (eps+mu) tuple; "
                    "the raw (...,3,3) array is not a supported layer material."
                )
        if not isinstance(formulation, Formulation):
            try:
                formulation = Formulation[formulation]
            except (KeyError, TypeError):
                raise RcwaError(
                    f"unknown formulation {formulation!r}; valid names: {', '.join(m for m in Formulation.__members__)}"
                ) from None
        self.formulation = formulation
        self.assume_periodic = True
        # Precompute the (design-INDEPENDENT) Fourier expansion ONCE, eagerly, host-side (the backend's
        # generate_expansion is pure numpy, cached here so the design solve is jit-safe).
        self._lv = LatticeVectors(u=np.array([period[0], 0.0]), v=np.array([0.0, period[1]]))
        self._ex = generate_expansion(self._lv, approximate_num_terms=orders)
        self._nt = self._ex.num_terms
        # The backend StateStack (expansion + chunking + guards fixed EAGERLY) is built as soon as a
        # concrete wavelength is known -- at construction when given, else on the first solve. The
        # stack's structural decisions never depend on the design, only the structure.
        self._stack = None
        self._prepared_grids = self._prepared_thick = self._prepared_wl = self._prepared_kin = None
        if wavelength is not None:
            self._build_stack(wavelength)

    @staticmethod
    def _ambient_scalar(e):
        """The scalar permittivity of a CONCRETE uniform isotropic ambient, or ``None`` when it is
        traced, in-plane patterned or anisotropic (the stage-2 ambient corner -- the analytic
        plane-wave split needs a structurally uniform, isotropic medium; anything else is loud or
        folded into the stack, never silently wrong).

        Accepts a scalar, a constant grid, or a 5/10-TUPLE whose components are all constant and
        represent an ISOTROPIC tensor (ε̂ = ε·I, μ̂ = I -- e.g. a σ=0 PML frame, which is the
        identity tensor); such a tuple reduces to its scalar ε exactly. A tuple with constant but
        genuinely ANISOTROPIC components is ``None`` too (a uniform-anisotropic ambient needs the
        general-anisotropic plane-wave split, folded at build like a patterned one)."""
        if isinstance(e, tuple):
            comps = [np.asarray(c) for c in e]
            # every component must be a constant grid (or scalar)
            vals = []
            for c in comps:
                a = c.reshape(-1)
                scale = max(float(np.abs(a).max()), 1.0)
                if np.ptp(a.real) > _UNIFORM_RTOL * scale or np.ptp(a.imag) > _UNIFORM_RTOL * scale:
                    return None
                vals.append(complex(a[0]))
            exx, exy, eyx, eyy, ezz = vals[:5]
            # isotropic tensor: eps_xx == eps_yy == eps_zz, zero off-diagonals
            if not (abs(exy) == 0.0 and abs(eyx) == 0.0 and exx == eyy and exx == ezz):
                return None
            if len(vals) == 10:  # mu must be the identity
                mxx, mxy, myx, myy, mzz = vals[5:]
                if not (abs(mxy) == 0.0 and abs(myx) == 0.0 and mxx == 1.0 and myy == 1.0 and mzz == 1.0):
                    return None
            return exx
        g = np.asarray(e)
        if g.ndim == 0:
            return complex(g)
        if g.size <= 1:
            return complex(g.reshape(-1)[0])
        scale = max(float(np.abs(g).max()), 1.0)
        if np.ptp(g.real) <= _UNIFORM_RTOL * scale and np.ptp(g.imag) <= _UNIFORM_RTOL * scale:
            return complex(g.reshape(-1)[0])
        return None

    @staticmethod
    def _is_pml_frame(e):
        """True when a 10-component tuple is a coordinate-stretch PML frame -- the ONLY 10-tuple the
        isolated-scatterer fold may absorb (see ``_build_stack``). ``None`` when a component is traced
        (cannot inspect; the front door only builds stretch frames, so the fold stays). ``False`` for
        any other 10-tuple.

        The front door builds the frame as ``eps^ = eps_rel·Λ``, ``mu^ = Λ`` (``Λ`` the diagonal
        coordinate-stretch tensor, ``eps_rel`` the physical permittivity, per pixel for a graded
        sigma): so a frame is exactly a DIAGONAL pair of tensors whose per-pixel ratio
        ``eps_ii/mu_ii`` is the SAME scalar across the three axes (the stretch enters both). A
        genuinely magnetic 10-tuple substrate (e.g. eps = (6,0,0,4,5), mu = (1,0,0,2,1.5)) has
        three different ratios (6, 2, 3.33) and is rejected here instead of silently becoming a
        unit slab over vacuum. Note: the in-plane ``mu_zz == mu_xx·mu_yy`` identity holds only at
        σ=0 (``Λ = diag(Sy/Sx, Sx/Sy, Sx·Sy)`` with ``Λ_xx·Λ_yy = 1``), so a real absorber frame
        does NOT satisfy it -- the common-ratio test above is the honest structural invariant."""
        if not isinstance(e, tuple) or len(e) != 10:
            return False
        if not all(_concrete(c) for c in e):
            return None  # traced: cannot inspect (documented seam limitation, like the other traced ambients)
        comps = [np.asarray(jnp.asarray(c)).reshape(-1) for c in e]
        exx, exy, eyx, eyy, ezz = comps[:5]
        mxx, mxy, myx, myy, mzz = comps[5:]
        scale = max(float(np.abs(np.concatenate([exx, eyy, ezz, mxx, myy, mzz])).max()), 1.0)
        tol = 1e-8 * scale
        # diagonal: off-diagonals ~ 0 relative to the frame scale
        for off in (exy, eyx, mxy, myx):
            if np.abs(off).max() > tol:
                return False
        # the stretch enters both: a common per-pixel ratio eps_ii/mu_ii across the three axes
        with np.errstate(divide="ignore", invalid="ignore"):
            r = [exx / mxx, eyy / myy, ezz / mzz]
        for i in range(3):
            for j in range(i + 1, 3):
                if not np.allclose(r[i], r[j], rtol=1e-6, atol=tol):
                    return False
        return True

    def _build_stack(self, wavelength):
        """Build the backend :class:`StateStack` from the CONSTRUCTION stack: the expansion, per-layer
        chunk counts, uniform/patterned flags and expm squaring guards are all decided here, eagerly,
        before any trace.

        Ambients (layers 0 and -1): a CONCRETE uniform isotropic ambient becomes the analytic
        plane-wave split scalar. A patterned or anisotropic-tensor ambient (the in-plane PML frame
        slices) CANNOT be an analytic scalar -- owning the slab it sits in is not the physics either
        -- so it is FOLDED into the stack as a finite layer of the jNO unit ambient thickness, with
        the analytic split against vacuum (epsilon = 1) beyond: the supercell behaves as an isolated
        scatterer in free space (exactly the in-plane PML's meaning, and how the fmmax-era seam
        treated these slice layers -- unit-thick eigensolved slabs)."""
        if not _concrete(wavelength):
            raise RcwaError(
                "the state-space backend needs a CONCRETE wavelength to fix its standing structure "
                "(chunking/squaring guards); solve with a concrete wavelength or pass wavelength= at "
                "construction."
            )
        sup = self._ambient_scalar(self.layers_spec[0][1])
        sub = self._ambient_scalar(self.layers_spec[-1][1])
        # A patterned / tensor ambient cannot be a scalar for the analytic plane-wave
        # split; fold the ambient slice into the stack as a unit ambient layer (the old
        # seam's inf -> 1.0 convention) and split against vacuum beyond (the PML supercell
        # is an isolated scatterer). The check is PER-SIDE: a patterned SCALAR-grid
        # ambient always raises loudly (task T30 corner) even when the other ambient is a
        # tuple, and a genuinely anisotropic CONSTANT 5-component (eps-only) ambient also
        # raises -- folding it would silently replace a semi-infinite birefringent
        # substrate by a 1-wavelength slab over air. Only the coordinate-stretch PML
        # frame (10-component eps^ + mu^ tensors, the stretch enters both) is folded,
        # whether the frame is patterned (sigma graded in-plane) or constant (uniform
        # sigma); that fold is the isolated-scatterer supercell model this code documents.
        fold_top, fold_bot = sup is None, sub is None
        for side, fold in ((0, fold_top), (-1, fold_bot)):
            if not fold:
                continue
            spec = self.layers_spec[side][1]
            if isinstance(spec, tuple) and len(spec) == 10:
                frame = self._is_pml_frame(spec)
                if frame is False:
                    raise RcwaError(
                        "a 10-component ambient is folded as an in-plane PML coordinate-stretch "
                        "frame ONLY when eps^ and mu^ are diagonal with the stretch entering both "
                        "(a common per-pixel ratio eps_ii/mu_ii across the axes -- the physical "
                        "permittivity). This 10-tuple is NOT a stretch frame (a genuine magnetic / "
                        "non-PML anisotropic substrate): folding it would silently model a "
                        "semi-infinite medium as one wavelength of slab over air. Use scalar/"
                        "constant-grid ambients on both sides, or replace the semi-infinite "
                        "ambient with a finite supercell."
                    )
                continue  # a validated PML coordinate-stretch frame (or a traced one) -- stays folded
            if isinstance(spec, tuple):
                # a 5-component eps-only uniform-anisotropic constant ambient
                raise RcwaError(
                    "a uniform-anisotropic constant 5-component ambient (eps^ only, mu = I) "
                    "is the stage-2 ambient corner: the analytic plane-wave split needs a "
                    "structurally uniform, isotropic medium, and the vacuum-beyond fold would "
                    "silently model a semi-infinite anisotropic substrate as one wavelength of "
                    "birefringent film over air. Only a 10-component coordinate-stretch PML "
                    "frame (eps^ + mu^) is folded; use scalar/constant-grid ambients on both "
                    "sides, or replace the semi-infinite ambient with a finite supercell."
                )
            # a patterned scalar grid ambient stays a loud stage-2 corner
            raise RcwaError(
                "a patterned scalar-grid ambient is the stage-2 ambient corner (task T30): the "
                "analytic plane-wave split needs a structurally uniform, isotropic medium; a "
                "patterned scalar grid is not supported (never silently wrong)."
            )
        films = []
        for k, (_t, e) in enumerate(self.layers_spec[1:-1]):
            t = _t
            if t is not None and not _concrete(t):
                raise RcwaError(
                    f"layer {k + 1} thickness is a traced JAX value: the eig-free state-space "
                    "backend fixes thickness at build (chunking depends on it), so a traced "
                    "thickness is not supported. Pass a concrete thickness."
                )
            if t is None or t == np.inf:  # jNO's historical mapping for a missing/ambient thickness
                t = 1.0
            if not (np.isreal(t) and t > 0):
                raise RcwaError(f"layer {k + 1} thickness must be > 0 (or None/inf for a ~unit ambient), got {t!r}.")
            films.append((float(t), e))
        if fold_top:
            films.insert(0, (1.0, self.layers_spec[0][1]))
        if fold_bot:
            films.append((1.0, self.layers_spec[-1][1]))
        self._fold_amb = (fold_top, fold_bot)
        dtype = jnp.dtype(jnp.complex128) if jax.config.jax_enable_x64 else jnp.dtype(jnp.complex64)
        try:
            self._stack = StateStack(
                wavelength=float(wavelength),
                period=self.period,
                amb_sup=1.0 if fold_top else sup,
                amb_sub=1.0 if fold_bot else sub,
                layers=films,
                orders=self.orders,
                k_in=self.k_in,
                formulation=self.formulation,
                dtype=dtype,
            )
        except NotImplementedError as e:  # a backend feature the state-space engine does not implement yet
            raise RcwaError(f"{e}") from e
        except ValueError as e:
            raise RcwaError(f"cannot build the state-space backend for this stack: {e}") from e

    def _prepare_stack(self, layers_spec, kin):
        """Each layer's permittivity material and the thickness list, EAGER on purpose: collapsing a
        uniform layer reads its values, which a trace cannot do -- so this runs before the compiled
        solve and hands it ``(1, 1)`` (the backend's uniform shortcut) or patterned grids as
        arguments. A layer material may be a scalar, a grid, or a 5/10-component anisotropic tuple
        (each component scalar or grid)."""
        kin = jnp.asarray(kin)
        cdt = jnp.result_type(kin.dtype, jnp.complex64)  # one complex dtype for the solve (robust to a
        for k, (_t, e) in enumerate(layers_spec):  # float32 design parameter vs float64 wavevectors)
            if isinstance(e, tuple):
                n = len(e)
                if n not in (5, 10):
                    raise RcwaError(
                        f"layer {k}: an anisotropic tuple material must have 5 (eps) or 10 (eps+mu) components, got {n}."
                    )
            elif jnp.ndim(e) >= 3:
                raise RcwaError(
                    f"layer {k}: a tensor-form ({tuple(jnp.shape(e))}) anisotropic layer must be "
                    "given as its 5-component (eps_xx..eps_zz) or 10-component (eps+mu) tuple; "
                    "the raw (...,3,3) array is not a supported layer material."
                )

        def _grid(g):  # scalar -> (1, 1) grid; jax-native so a permittivity grid flows -> differentiable in ε
            g = jnp.asarray(g).astype(cdt)
            return jnp.full((1, 1), g) if g.ndim == 0 else g

        def _collapse(g):
            """Collapse a CONCRETE constant grid to ``(1, 1)`` so the backend takes its ANALYTIC
            uniform path (exact plane waves, no expm) instead of the patterned (4n) transfer.
            ``detect_layers`` hands back a full ``(Ng, Ng)`` grid for EVERY slab, so constant
            cladding / substrate / mirror films must collapse -- ruinous otherwise for a
            mostly-uniform stack (a Bragg / Mo/Si multilayer). Exact, not an approximation. A
            TRACED grid never collapses: under trace a "uniform" grid may be a design parameter
            that merely starts flat, and collapsing it would freeze the design's structure out of
            the solve."""
            if _concrete(g) and g.size > 1:
                a = np.asarray(g)
                scale = max(float(np.abs(a).max()), 1.0)
                if np.ptp(a.real) <= _UNIFORM_RTOL * scale and np.ptp(a.imag) <= _UNIFORM_RTOL * scale:
                    return jnp.full((1, 1), a.reshape(-1)[0], dtype=cdt)
            return g

        def prepare(e):
            if isinstance(e, tuple):
                comps = [_grid(c) for c in e]
                # ALL-OR-NOTHING across the layer's components: a tensor layer collapses to
                # uniform only when every component is concrete-constant -- a partly-uniform
                # tensor must stay fully patterned (mirrors the old seam / fmmax shape rule:
                # the backend's Fourier matrices need matching component shapes).
                collapsed = [_collapse(c) for c in comps]
                if all(tuple(jnp.shape(c)) == (1, 1) for c in collapsed):
                    return tuple(collapsed)
                return tuple(_grid(c) for c in e)
            return _collapse(_grid(e))

        grids = [prepare(e) for _, e in layers_spec]
        for k, (t, _) in enumerate(layers_spec):
            # A TRACED per-solve thickness hits `t == np.inf` below and raises a raw
            # TracerBoolConversionError. The eig-free backend fixes thickness at build
            # (chunking and the expm squaring guards depend on it), so a traced thickness
            # is an API regression vs main -- raise the same loud RcwaError as the
            # concrete-mismatch branch instead of a confusing tracer crash.
            if t is not None and not _concrete(t):
                raise RcwaError(
                    f"per-solve thickness for layer {k} is a traced JAX value: the eig-free "
                    "state-space backend fixes thickness at build (chunking depends on it), so a "
                    "traced thickness is not supported. Pass a concrete thickness, or create a "
                    "fresh Rcwa for a new thickness."
                )
        thick = [jnp.asarray(1.0 if t is None or t == np.inf else t) for t, _ in layers_spec]
        return grids, thick

    def _backend_materials(self, grids):
        """The (films, amb_sup_scalar, amb_sub_scalar) handed to the StateStack core, EAGERLY.

        With a folded ambient (in-plane PML: the grid at 0 or -1 is a patterned/tensor slice), the
        ambient grid BECOMES the outermost film and the analytic split scalar is vacuum (= 1) beyond
        it; otherwise the ambient is the uniform (1,1)/scalar medium and ``_amb`` reads its corner
        value (jax-native, trace-safe). A TUPLE ambient (uniform-anisotropic tensor, validated
        isotropic-constant at build) reduces to its diagonal value ``ε_xx``."""
        fold_top, fold_bot = getattr(self, "_fold_amb", (False, False))

        def _amb(g):
            if isinstance(g, tuple):
                g0 = jnp.asarray(g[0])  # eps_xx component; isotropic tensor => the diagonal value
                return g0 if g0.ndim == 0 else g0[0, 0]
            g = jnp.asarray(g)
            return g if g.ndim == 0 else g[0, 0]

        films = []
        for i, g in enumerate(grids):
            is_amb = (i == 0 and not fold_top) or (i == len(grids) - 1 and not fold_bot)
            if not is_amb:
                films.append(g)
        sup = 1.0 if fold_top else _amb(grids[0])
        sub = 1.0 if fold_bot else _amb(grids[-1])
        return tuple(films), sup, sub

    def _ensure_stack(self, grids, thick, wl):
        """Lazy engine build + per-solve structure validation (count/shapes/thicknesses are fixed at
        build; a mismatch raises loudly instead of silently solving a different structure)."""
        if self._stack is None:
            self._build_stack(wl)
        st = self._stack
        # Degenerate ambient-only stacks (0 finite films: a single homogenized medium, or a bare
        # super|sub pair) are legal -- the grids are the ambients, exactly as the fmmax-era engine
        # treated them (T = 1 trivially when the two sides share one medium).
        films, _, _ = self._backend_materials(grids)
        n_film = max(0, len(films))
        if n_film != len(st.layers):
            raise RcwaError(
                f"per-solve layer stack has {n_film} finite layers but the engine was built with "
                f"{len(st.layers)} -- the state-space backend fixes the structure at build; create a "
                "fresh Rcwa for a different structure."
            )
        # The analytic plane-wave split needs UNIFORM ambients. `_prepare_stack` collapses a concrete
        # constant ambient grid to (1, 1), so a remaining CONCRETE non-(1, 1) ambient grid here is a
        # genuinely patterned grid -- and the compiled seam's `_amb` would slice its corner value
        # g[0, 0] and silently solve that scalar as the whole ambient (the stage-2 ambient corner,
        # task T30). A TRACED ambient grid is allowed: it arises only from the parametric engine's
        # resample, whose ambients are scalar-derived and therefore uniform at every trace point
        # (corner-slice exact); a traced-*patterned* ambient cannot be checked eagerly (values are
        # invisible under trace) and is a documented stage-2 limitation.
        fold_top, fold_bot = getattr(self, "_fold_amb", (False, False))
        for gi, folded in ((0, fold_top), (len(grids) - 1, fold_bot)):
            if folded:
                continue  # this ambient is a folded STACK film, not an analytic-split ambient
            g = grids[gi]
            if isinstance(g, tuple):
                # A per-solve TUPLE ambient reaches an analytic-split side. `_ambient_scalar`
                # validated the BUILD-time ambient; a per-solve override is NOT re-validated
                # by anything (the fold flags were fixed at build), and the compiled seam's
                # `_amb` corner-slices g[0][0, 0] = eps_xx for a whole tensor. So a per-solve
                # tuple is allowed only when it reduces EXACTLY to its scalar (an
                # isotropic-constant tensor: the corner slice is then the whole physics);
                # a genuinely anisotropic per-solve tuple would be silently solved as an
                # eps_xx isotrope -- the never-silent violation -- and raises loudly.
                # (A TRACED per-solve tuple cannot be inspected eagerly: document the joint
                # limitation alongside the traced-*patterned* ambient note above.)
                if all(_concrete(c) for c in g):
                    flat = tuple(np.asarray(jnp.asarray(c)) for c in g)
                    if self._ambient_scalar(flat) is None:
                        raise RcwaError(
                            f"per-solve ambient {gi} is a genuinely anisotropic tensor tuple: the "
                            "analytic plane-wave split was fixed at build for THIS stack's ambient "
                            "kind, and the compiled seam would corner-slice it to eps_xx (never "
                            "silently wrong). Pass the anisotropic tuples at construction (the "
                            "stage-2 ambient corner, task T30), or create a fresh Rcwa."
                        )
                continue  # isotropic-constant tuple: the corner-slice is exact
            gshape = tuple(jnp.shape(g))
            if _concrete(g) and gshape not in ((), (1, 1)):
                raise RcwaError(
                    f"ambient layer {gi}'s per-solve grid shape {gshape} is not a uniform (1,1) "
                    "permittivity -- the analytic plane-wave split needs a structurally uniform "
                    "ambient; a patterned scalar-grid ambient is the stage-2 corner (task T30) and "
                    "raises here instead of returning a wrong answer."
                )
        # The backend's build-time per-layer material shape may be exposed as the
        # scalar-layer `_shapes` (stage-1) or, after the anisotropic refactor, the
        # per-component `_comp_shapes` (one entry per layer, itself a tuple). Read
        # whichever the backend exposes so the structure-freeze check works across
        # both: a per-solve scalar-isotropic grid must match the (first) component.
        if hasattr(st, "_shapes"):
            build_shapes = list(st._shapes)
        else:
            build_shapes = [cs[0] for cs in st._comp_shapes]
        for li, g in enumerate(films):
            if isinstance(g, tuple):
                gshape = tuple(tuple(jnp.shape(c)) for c in g)
                bshape = st._comp_shapes[li]
                ok = gshape == bshape or all(s == (1, 1) for s in gshape)
                if not ok:
                    raise RcwaError(
                        f"layer {li + 1} material shapes {gshape} differ from the engine's "
                        f"{bshape} (structure fixed at build) -- create a fresh Rcwa."
                    )
                continue
            gshape = tuple(jnp.shape(g))
            # A per-solve constant grid is collapsed to (1, 1) eagerly (exact): accept it for a
            # genuinely patterned build shape too -- the uniform shortcut is a per-solve property.
            if gshape != build_shapes[li] and gshape != (1, 1):
                raise RcwaError(
                    f"layer {li + 1} grid shape {gshape} differs from the engine's "
                    f"{build_shapes[li]} (structure fixed at build) -- create a fresh Rcwa."
                )
        for li, t in enumerate(thick[1:-1]):
            # The built thickness may live in float32/complex64 storage (e.g. stored as a complex64
            # scalar) or be reached through a float32 per-solve JAX array: 0.35 -> 0.35000002 in
            # float32, so a 1e-9 comparison against the exact float64 value fails spuriously. Compare
            # with a float32-safe relative tolerance (float32 eps ~1.2e-7, so 1e-6 rel. gives ~10x
            # margin) while keeping the guard's spirit: a genuinely different thickness still raises.
            # With FOLDED ambients (in-plane PML) the engine inserts the ambient slices as the
            # outermost films, so film index == full-layer index (``li + 1`` here); without a fold
            # the films are the central slices, shifted by one from the full grid.
            f = (li + 1) if fold_top else li
            tb = float(np.real(st.layers[f][0]))
            if _concrete(t) and abs(float(t) - tb) > 1e-6 * max(1.0, abs(tb)):
                raise RcwaError(
                    f"per-solve thickness {float(t):.6g} for layer {li + 1} differs from the engine's "
                    f"built {tb:.6g}; chunking is fixed at build -- create a fresh Rcwa for a new "
                    "thickness."
                )
        return st

    def _solve_core(self, grids, thick, wl, kin, pol):
        """Backend layer solves + chunked R-matrix stack assembly + incident state/flux at k_in -- the
        numerical core of a plane-wave solve, COMPILED once per stack structure (the ``_core_jit``
        cache, exactly as before, so re-solves across wavelength/design/k_in/pol reuse the executable).
        Structural decisions (collapse, chunking, squaring guards, the ambient fold) stay outside,
        eager."""
        fn = self.__dict__.get("_core_jit")
        if fn is None:
            stack = self._stack

            def core(grids, thick, wl, kin, pol):
                films, sup, sub = self._backend_materials(grids)
                return stack._solve_core(films, wl, kin, pol, sup, sub)

            fn = self._core_jit = jax.jit(core)
        return _Result.from_tuple(
            fn(tuple(grids), tuple(thick), jnp.asarray(wl), jnp.asarray(kin), tuple(jnp.asarray(p) for p in pol))
        )

    def _resolve_pol(self, pol):
        """Re-solve the last solve's stack at an alternate input polarization (``jones`` / the
        vector-aerial pupil) -- through the SAME compiled core, so it is a cheap reuse."""
        return self._solve_core(
            self._prepared_grids, self._prepared_thick, self._prepared_wl, self._prepared_kin, _as_jones(pol)
        )

    def solve(self, *, wavelength=None, k_in=None, layers=None, source=None, polarization=None):
        """Solve the stack and return a :class:`_Sol`. Raises if the wavelength is unknown or energy is
        not conserved.

        ``layers`` optionally overrides the construction layer stack with ``[(thickness, eps), ...]`` at
        solve time -- pass JAX permittivity grids here to differentiate the solve in the design (the
        construction-time shape guards run once, eagerly, so the solve itself stays trace-clean).

        Thickness is STRUCTURAL: the eig-free backend fixes each layer's thickness at build (the chunked
        R-matrix recursion and its expm squaring guards depend on it), so per-solve thicknesses must be
        CONCRETE and equal to the built ones. A traced thickness on ``solve(layers=...)`` raises
        ``RcwaError`` (differentiable thickness is not supported by the eig-free backend; create a fresh
        ``Rcwa`` for a new thickness).

        ``source`` (a dict built by the front door) drives an **internal-source** emission solve instead of
        plane-wave incidence -- see :meth:`_solve_source`; it returns an :class:`_EmitterSol`.

        Args:
            wavelength: free-space wavelength; overrides the construction value.
            k_in: transverse incident wavevector; overrides the construction value.
            layers: ``[(thickness, eps), ...]`` replacing the stack for this solve (JAX arrays allowed).
            source: an internal-source description (built by :func:`jno.rcwa`); switches to an emission solve.
            polarization: incident polarization override.

        Returns:
            The solution: efficiencies (``.efficiency("T")``, ``.order(m, n)``) or, with ``source``, the
            emitted powers.

        Raises:
            RcwaError: if the wavelength is unknown or energy is not conserved.
        """
        wl = wavelength if wavelength is not None else self.wavelength
        if wl is None:
            raise RcwaError(
                "wavelength is unknown: pass wavelength= to solve() or at construction; it sets every "
                "layer's state transfer and is never defaulted."
            )
        layers_spec = self.layers_spec if layers is None else layers
        kin = jnp.asarray(self.k_in if k_in is None else k_in, float)  # jax -> incidence angle is differentiable
        nt = self._nt  # precomputed eagerly in __init__ (jit-safe)
        if source is not None:
            return self._solve_source(source, layers_spec, wl, kin)
        pol = self.polarization if polarization is None else _as_jones(polarization)
        grids, thick = self._prepare_stack(layers_spec, kin)
        self._ensure_stack(grids, thick, wl)  # lazy engine build + structure validation
        self._prepared_grids, self._prepared_thick = tuple(grids), tuple(thick)
        self._prepared_wl, self._prepared_kin = jnp.asarray(wl), jnp.asarray(kin)
        with _annot("rcwa:layer_solves+stack"):
            res = self._solve_core(grids, thick, wl, kin, pol)
        if _concrete(res.pin) and float(res.pin) <= 0:
            raise RcwaError("no forward-propagating incident mode in the superstrate; check wavelength/period.")
        sol = _Sol(self, res, self._ex, nt, res.pin, wl, thick=thick, period=self.period, pol=pol)
        T, R = sol.efficiency("T"), sol.efficiency("R")
        if _concrete(T) and _concrete(R):  # eager: never-silent, raise
            msg = _efficiency_problem(T, R, self.orders, self.period, wl)
            if msg is not None:
                raise RcwaError(msg)
        else:  # under jit/grad/vmap: the same check on the host, as a warning
            wl_shown = float(wl) if _concrete(wl) else "traced"
            jax.debug.callback(
                functools.partial(_warn_on_efficiency_problem, orders=self.orders, period=self.period, wl=wl_shown),
                jnp.real(T),
                jnp.real(R),
            )
        return sol

    def _solve_source(self, source, layers_spec, wl, kin):
        """Internal-source (dipole / Gaussian) emission solve. ``source`` is a dict from the front door
        (or the same shape at engine level)::

            {loc: (1,2) JAX, layer: int (full-stack index incl. ambients),
             t_upper, t_lower: JAX, kind: 'delta'|'gaussian', fwhm: JAX,
             orient: (ox,oy,oz), amp: scalar}

        The backend's :func:`jno._rcwa.sources.emission_solve` closes the Maxwell jump conditions
        (H_t jumps by j_t, E_t continuous; the z-current couples through the inverse-z-permittivity)
        at the source plane between the two half-stacks -- the eig-free substitute for fmmax's
        ``amplitudes_for_source``. Returns an :class:`_EmitterSol` with the power radiated up / down
        (differentiable in ``amp``, ``orient``, ``loc``, ``t_upper``/``t_lower`` and the design ε)."""
        from jno._rcwa import sources as _r_sources

        grids, thick = self._prepare_stack(layers_spec, kin)
        st = self._ensure_stack(grids, thick, wl)  # lazy engine build + structure validation

        # The front door's `layer` indexes the FULL layer stack (ambients at 0 and -1, top-first);
        # the backend StateStack holds only the finite films. Without a top fold the backend films
        # are the central slices (subtract the top ambient); WITH a top fold the folded ambient
        # slice became films[0], so a full-stack index k >= 1 maps to backend film k unchanged --
        # the shift must be fold-aware (never the bare -1: under a PML frame it would silently
        # place the source one film off).
        fold_top, _ = getattr(self, "_fold_amb", (False, False))
        if not isinstance(source["layer"], (int, np.integer)):
            raise RcwaError(f"source layer must be an integer stack index, got {source['layer']!r}.")
        src = dict(source)
        src["layer"] = int(source["layer"]) - (0 if fold_top else 1)
        try:
            # Wire through the SAME shared _backend_materials path as the plane-wave core: the
            # (films, sup, sub) triple already accounts for folded ambients (the folded slice
            # lives inside `films`, and the analytic split scalar is vacuum beyond) and for
            # tuple ambients (the validated isotropic-constant corner slice). A hand-rolled
            # local `_amb` and a bare grids[1:-1] here would under-count backend layers under a
            # fold (the "N entries for N+1 backend layers" misattribution) and silently slice a
            # tuple ambient to eps_xx.
            films, sup, sub = self._backend_materials(grids)
            up, down = _r_sources.emission_solve(
                st,
                src,
                wavelength=wl,
                eps_grids=tuple(films),
                k_in=kin,
                amb_sup=sup,
                amb_sub=sub,
            )
        except ValueError as e:
            raise RcwaError(str(e)) from e
        if _concrete(up) and not (np.isfinite(float(up)) and np.isfinite(float(down))):
            raise RcwaError(
                "non-finite emission (NaN/inf): the internal-source solve hit a Γ-point / Rayleigh "
                f"degeneracy (k_in={tuple(float(k) for k in kin)}). Nudge k_in off exact normal, "
                "e.g. k_in=(1e-3, 1e-3)."
            )
        return _EmitterSol(up, down)


# =====================================================================================
# Front door: infer an RCWA problem from a jNO constraint list / FEM problem + eps.
# =====================================================================================
@dataclass
class RcwaSpec:
    """Everything the RCWA engine needs, inferred from a jNO problem. Inspectable with no RCWA
    backend pulled in.

    Attributes:
        period: lattice periods ``(Lx, Ly)``.
        layers: the stack, ``[(thickness, eps_xy), ...]``.
        wavelength: free-space wavelength.
        k_in: transverse incident wavevector ``(kx, ky)``.
        source_face: the illuminated face's tag.
        ambient_faces: the tags of the two ambient (radiating) faces.
        periodic_axes: the Floquet-tied axes and their face pairs.
        source: an internal-source (dipole/Gaussian) emission spec, or ``None`` for plane-wave incidence.
        polarization: incident transverse Jones vector ``(p_x, p_y)``, unit norm.
        polarization_from: how the polarization was decided (given, read from a vector source, or inferred).
    """

    period: tuple
    layers: list  # [(thickness, eps_xy), ...]
    wavelength: float
    k_in: tuple = (0.0, 0.0)
    source_face: str = ""
    ambient_faces: tuple = ()
    periodic_axes: dict = field(default_factory=dict)
    source: dict = None  # an internal-source (dipole/Gaussian) emission spec, else None (plane-wave incidence)
    polarization: tuple = (1.0 + 0j, 0j)  # incident transverse Jones vector (px, py), unit norm
    polarization_from: str = ""  # how it was decided -- given, read from a vector source, or inferred

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
            ties.append((spec[0], spec[1]))  # (main, secondary) tags
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
    tol = 1e-6 * max(float(np.ptp(pts, axis=0).max()), 1.0)
    for tag, br in getattr(domain, "_boundary_regions", {}).items():
        if tag == "boundary" or br is None or len(getattr(br, "points", [])) == 0:
            continue
        bz = np.asarray(br.points)[:, 2]
        if not (np.all(np.abs(bz - zmin) < tol) or np.all(np.abs(bz - zmax) < tol)):
            continue  # an ambient face lies in the bottom or top plane; interfaces and side faces do not
        if tag not in normals:
            # A MISSING normal used to read as a zero vector, and `0 < 0.8*(0+1e-30)` is True — so an
            # untagged-normal face was silently *disqualified* and the search failed with "could not
            # identify the ambient faces", naming neither the tag nor the real cause. Say it instead.
            raise RcwaError(
                f"boundary region {tag!r} carries no per-point normals (`domain.normals_by_tag` has no "
                f"entry for it), so RCWA cannot tell whether it is a z-normal ambient face. Tag it with "
                "a coordinate or facet predicate on a meshed boundary so its normals are derived."
            )
        nrm = np.asarray(normals[tag]).mean(0)
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


def _pml_layer_components(mass_node, stretch, period, grid, zmid, params, k0sq, sub=1):
    """The 10 relative-permittivity/permeability grids ``(ε_xx,ε_xy,ε_yx,ε_yy,ε_zz, μ_xx,μ_xy,μ_yx,μ_yy,μ_zz)``
    of an in-plane uniaxial PML layer at height ``zmid``.

    ``Λ = diag(c_xx, c_yy, c_zz)`` is read off the stiffness coefficients (``stretch``); the Maxwell PML is
    ``ε̂ = ε·Λ``, ``μ̂ = Λ`` with ``ε = (K0²·ε·SₓSᵧS_z)/k0² / (c_xx·c_yy·c_zz)`` since
    ``SₓSᵧS_z = c_xx·c_yy·c_zz``. Off-diagonal components are zero (uniaxial). ``sub`` = subpixel smoothing
    (each ε̂/μ̂ component is built at ``grid·sub`` and averaged per pixel)."""
    g = grid * sub
    pts = _cell_grid_at_z(period, g, zmid)

    def samp(node):
        v = jnp.full((g * g,), node) if np.isscalar(node) else _eval_coeff_points(node, pts, params)
        return jnp.reshape(v, (g, g)) + 0j

    lam = [samp(stretch[ax]) for ax in (0, 1, 2)]
    m = jnp.reshape(_eval_coeff_points(mass_node, pts, params), (g, g)) + 0j
    eps_rel = (m / k0sq) / (lam[0] * lam[1] * lam[2])
    z = jnp.zeros((g, g), complex)
    comps = (eps_rel * lam[0], z, z, eps_rel * lam[1], eps_rel * lam[2], lam[0], z, z, lam[1], lam[2])
    return tuple(_pixel_average(c, grid, sub) for c in comps)


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


def _cell_centres(length, n):
    """The ``n`` sample positions of one period ``[0, length)``: the CENTRES of its ``n`` equal cells.

    This is the convention of the Fourier transform the layers go through (fmmax places array element
    ``j`` at ``(j + 1/2) * length / n``), so the Fourier coefficients carry the right phase and a material
    edge at a simple fraction of the period (``y < 0.5 * period``) never lands exactly on a sample, where
    floating-point rounding would decide its side and move the edge by half a cell (a first-order error)."""
    return (np.arange(n) + 0.5) * (length / n)


def _localize_source(src_node, is_vector, period, z_range, grid, nz, params):
    """Locate an internal source by its ``|·|²`` distribution on the cell volume: the centroid ``(x₀,y₀,z₀)``
    and lateral width → **point** (``dirac_delta_source``) vs **Gaussian** (``gaussian_source`` with a fwhm).
    Returns ``dict(x0, y0, z0, kind, fwhm)`` (all static geometry; the amplitude flows separately)."""
    xs = _cell_centres(period[0], grid)
    ys = _cell_centres(period[1], grid)
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
    from jno.jnp_ops import cross as jno_cross
    from jno.trace import FunctionCall, TestFunction

    def _vector_load_source(f):
        """A factor ``inner(g, v×n)`` (trial-free ``g``) -> the tangential forcing it applies, ``n×g`` (so
        ``inner(cross(E_inc, n), v×n)`` gives ``E_inc,t``, the incident polarization); ``n×v`` flips the
        sign. ``None`` for any other factor."""
        from jno.utils.solver.fem_nonnodal import _n1e_tangential_cross

        if not (isinstance(f, FunctionCall) and getattr(f, "_name", None) == "inner" and len(f.args) == 2):
            return None
        a0, a1 = f.args
        for g, tr in ((a0, a1), (a1, a0)):
            tv = _n1e_tangential_cross(tr, TestFunction)
            if tv is not None and not (_contains_trial(g) or _contains_test(g)):
                nvar = tr.args[1] if tv[0] > 0 else tr.args[0]
                force = jno_cross(nvar, g)
                return force if tv[0] > 0 else (-1.0) * force
        return None

    expr = getattr(term, "expr", term)
    parts = []
    for tsign, summand in _add_split(expr):
        fac = _mul_factors(summand)
        # vector Maxwell incident source: a factor inner(g, v×n); source = (other factors) · n×g
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

    Works for a scalar source (Helmholtz) and a VECTOR source (Maxwell, ``inner(cross(E_inc, n), v×n)``): a plane wave is
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


def _pixel_average(vals, grid, sub):
    """**Subpixel smoothing**: area-average a ``(grid·sub, grid·sub[, …])`` supersampled field down to
    ``(grid, grid[, …])`` by averaging each pixel's ``sub×sub`` block. Anti-aliases material boundaries so
    the Fourier expansion converges faster (less Gibbs ringing) and the gradient w.r.t. a boundary-moving
    design parameter is smooth, not staircased. A plain mean -- differentiable, works on NumPy or JAX."""
    if sub == 1:
        return vals
    tail = tuple(vals.shape[2:])
    return vals.reshape(grid, sub, grid, sub, *tail).mean(axis=(1, 3))


def _sample_grid_direct(coeff_node, grid, nz, period, z_range, params=None, sub=1):
    """Evaluate an analytic coefficient expression directly on the RCWA grid (exact, no mesh noise).

    Used when the permittivity is an analytic function of the coordinates (no per-node field); this
    avoids the staircase/interp noise that makes z-slices look non-invariant on an unstructured mesh.
    ``params`` substitutes trainable-parameter values (so the eager structure/k0 inference uses valid
    values, e.g. a K0 parameter). ``sub`` supersamples each pixel ``sub×sub`` and averages (subpixel
    smoothing)."""
    g = grid * sub
    xs = _cell_centres(period[0], g)
    ys = _cell_centres(period[1], g)
    zs = np.linspace(z_range[0], z_range[1], nz)
    GX, GY, GZ = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1)
    vals = np.asarray(_eval_coeff_points(coeff_node, pts, params or {}))
    if vals.ndim >= 3 and vals.shape[-2:] == (3, 3):  # anisotropic ε̂ -> layer-detect on the xx component
        vals = vals[..., 0, 0]
    vals = vals.reshape(g, g, nz)
    if sub > 1:  # anti-alias: average each pixel's sub×sub supersamples (z untouched)
        vals = vals.reshape(grid, sub, grid, sub, nz).mean(axis=(1, 3))
    return np.moveaxis(vals, 2, 0).astype(complex), zs  # -> (nz, grid, grid)


def _interface_bisection(coeff_node, grid, period, params, sub=1, iters=60):
    """``locate(z_lo, z_hi)`` for :func:`detect_layers`: bisect the height of the interface between two
    layers on the analytic permittivity, to ``(z_hi - z_lo) / 2**iters`` -- machine precision.

    Each probe samples the whole cell plane at one height and asks which of the two neighbouring slabs it
    matches, so a pattern that changes at the interface (a pillar ending) is located as exactly as a
    uniform film. Exact interfaces matter: RCWA's answer is a Fabry-Perot sum over the layer thicknesses."""
    g = grid * sub

    def plane(z):
        vals = np.asarray(_eval_coeff_points(coeff_node, _cell_grid_at_z(period, g, z), params or {}))
        if vals.ndim >= 3 and vals.shape[-2:] == (3, 3):
            vals = vals[..., 0, 0]
        return _pixel_average(vals.reshape(g, g), grid, sub)

    def locate(lo, hi):
        below, above = plane(lo), plane(hi)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            p = plane(mid)
            if np.max(np.abs(p - below)) <= np.max(np.abs(p - above)):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    return locate


def _interface_bisection_nodal(points, values, grid, period, sub=1, iters=40):
    """``locate(z_lo, z_hi)`` for a MESH-FIELD permittivity: bisect on its piecewise-linear interpolant
    (a Delaunay triangulation of the nodes, as the differentiable re-sampling uses), so the interface is
    where the field crosses halfway between the two layers -- the geometry the mesh actually represents,
    to within its own resolution. A slab that falls between mesh nodes cannot be located better than that."""
    from scipy.spatial import Delaunay, cKDTree

    tri, tree = Delaunay(points), cKDTree(points)
    vals_n = np.asarray(values).reshape(-1)
    g = grid * sub

    def plane(z):
        q = _cell_grid_at_z(period, g, z)
        s = tri.find_simplex(q)
        out = np.empty(len(q), vals_n.dtype)
        inside = s >= 0
        if inside.any():
            T = tri.transform[s[inside]]
            b3 = np.einsum("kij,kj->ki", T[:, :3, :], q[inside] - T[:, 3, :])
            w = np.c_[b3, 1.0 - b3.sum(axis=1)]
            out[inside] = np.sum(w * vals_n[tri.simplices[s[inside]]], axis=1)
        if (~inside).any():
            out[~inside] = vals_n[tree.query(q[~inside])[1]]
        return _pixel_average(out.reshape(g, g), grid, sub)

    def locate(lo, hi):
        below, above = plane(lo), plane(hi)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            p = plane(mid)
            if np.max(np.abs(p - below)) <= np.max(np.abs(p - above)):
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    return locate


def _coeff_is_tensor(coeff_node, period, z_range, params):
    """True if the permittivity coefficient is a 3×3 tensor (anisotropic ε̂) rather than a scalar — decided
    by evaluating it once at a representative cell point and checking the value shape."""
    pt = np.array([[period[0] * 0.5, period[1] * 0.5, 0.5 * (z_range[0] + z_range[1])]])
    v = np.asarray(_eval_coeff_points(coeff_node, pt, params or {}))
    return v.ndim >= 3 and v.shape[-2:] == (3, 3)


def _tensor_components(coeff_node, period, grid, zmid, params, k0sq, sub=1):
    """The 5 relative-permittivity component grids (ε_xx, ε_xy, ε_yx, ε_yy, ε_zz) the backend needs, sampled on the
    unit-cell grid at height ``zmid`` from the K0²·ε̂ tensor coefficient (divided by k0²). ``sub`` = subpixel
    smoothing factor."""
    g = grid * sub
    T = jnp.reshape(_eval_coeff_points(coeff_node, _cell_grid_at_z(period, g, zmid), params), (g, g, 3, 3))
    T = _pixel_average(T, grid, sub) / k0sq
    return (T[..., 0, 0], T[..., 0, 1], T[..., 1, 0], T[..., 1, 1], T[..., 2, 2])


def _sample_grid(domain, eps_nodes, grid, nz, period, z_range, sub=1):
    """Interpolate nodal eps onto an (nz, grid, grid) unit-cell array over the z-extent (``sub`` = subpixel
    supersampling, averaged per pixel)."""
    from scipy.interpolate import griddata

    g = grid * sub
    pts = np.asarray(domain.points)
    xs = _cell_centres(period[0], g)
    ys = _cell_centres(period[1], g)
    zs = np.linspace(z_range[0], z_range[1], nz)
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    E = np.empty((nz, grid, grid), complex)
    for k, zz in enumerate(zs):
        sel = np.abs(pts[:, 2] - zz) < (z_range[1] - z_range[0]) / (nz - 1) * 0.75 + 1e-9
        if sel.sum() < 3:
            j = np.argsort(np.abs(pts[:, 2] - zz))[: max(3, g)]
            sel = np.zeros(len(pts), bool)
            sel[j] = True
        vals = griddata(pts[sel][:, :2], eps_nodes[sel], (GX, GY), method="nearest")
        E[k] = vals.reshape(grid, sub, grid, sub).mean(axis=(1, 3)) if sub > 1 else vals
    return E, zs


def _source_dof_coords(domain, n_dof):
    """Coordinates the flat solution/forcing DOFs live on: mesh vertices for a nodal (Lagrange) field,
    or edge midpoints for a Nédélec (N1E) edge field, whose DOFs are edges — so the source-face detection
    reads the right positions for a vector Maxwell problem."""
    pts = np.asarray(domain.points)
    if n_dof == len(pts):
        return pts  # nodal (scalar Helmholtz)
    _topo = getattr(domain, "_fem_nonnodal_topology", None) or {}
    _dm = _topo.get("dofmap")
    if _dm is not None and int(_dm.n_dofs) == n_dof and _dm.degree > 1:
        # a degree-k H(curl) field: each DOF sits on an edge, face or cell -- use that entity's centroid
        return _dm.dof_entity_centroids(pts[:, : _dm.tdim])
    from jno.utils.solver.fem_topology import BASIX_TET_EDGES, BASIX_TRIANGLE_EDGES, build_edge_topology

    dim = getattr(domain, "dimension", 3)
    cells = np.asarray(domain.mesh.cells_dict["tetra" if dim == 3 else "triangle"])
    topo = build_edge_topology(cells, BASIX_TET_EDGES if dim == 3 else BASIX_TRIANGLE_EDGES)
    if topo.n_edges == n_dof:  # N1E edge DOFs -> edge midpoints
        return pts[np.asarray(topo.edge_vertices)].mean(axis=1)
    raise RcwaError(
        f"cannot map the {n_dof} forcing DOFs to coordinates (mesh has {len(pts)} vertices, "
        f"{topo.n_edges} edges); RCWA source-face detection supports nodal (Lagrange) and N1E (any degree) fields."
    )


def _source_kin(femobj, domain, params=None):
    """Read the illuminated face and transverse wavevector k_in from the assembled forcing b.
    ``params`` overrides trainable-parameter values used when assembling (e.g. a K0/source parameter)."""
    # A complex form is FUSED into one real 2n system at assembly, so ``fem.operator`` is that block and
    # no longer the Re/Im pair. The unfused legs are retained on ``_complex_legs`` for exactly the
    # consumers that still want them (here, and the complex-native AMS preconditioner) — read those
    # first, and fall back to ``.operator`` for the un-fused Bloch remnant, whose operator IS the pair.
    op = getattr(femobj, "_complex_legs", None) or femobj.operator
    if not (isinstance(op, tuple) and len(op) == 2):
        raise RcwaError(
            "jno.rcwa expects a complex (time-harmonic) Helmholtz problem: neither fem._complex_legs nor "
            "fem.operator gave a (real, imag) operator pair. Author the field with complex=True."
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
    xs = _cell_centres(period[0], grid)
    ys = _cell_centres(period[1], grid)
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


def _flat_face_normal(domain, tag):
    """Outward unit normal of the flat boundary face ``tag``: the normal of the plane through its points
    (least squares), pointing away from the mesh centroid. Raises if the face is not flat."""
    fp = np.asarray(_boundary_points(domain, tag), dtype=float)
    c = fp.mean(0)
    _, sv, vt = np.linalg.svd(fp - c, full_matrices=False)
    if sv[-1] > 1e-8 * max(1.0, sv[0]):
        raise RcwaError(f"the source reads the normal of face {tag!r}, which is not flat.")
    n = vt[-1]
    return n if np.dot(c - np.asarray(domain.mesh.points, dtype=float)[:, : len(n)].mean(0), n) > 0 else -n


def _boundary_points(domain, tag):
    br = getattr(domain, "_boundary_regions", {}).get(tag)
    if br is None or len(getattr(br, "points", [])) == 0:
        raise RcwaError(f"boundary region {tag!r} has no points; cannot locate it.")
    return br.points


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
    from jno._fem import _is_normal_variable
    from jno.trace import RegionMask, region_mask_key

    pts = jnp.asarray(pts)
    ctx = {}
    for v in _walk_nodes(coeff_node):
        if isinstance(v, RegionMask):
            # a region-attached coefficient (`.attach(eps=...)`): membership of the sample points, by the
            # same geometric rule the FEM assembly applies to cell centroids
            from jno.utils.solver.fem_utils import region_membership_at

            dom = getattr(v, "_domain", None)
            m = None if dom is None else region_membership_at(dom, v.region, np.asarray(pts)[:, : int(dom.dimension)])
            if m is None:
                raise RcwaError(
                    f"the permittivity reads region {v.region!r}, which has no point membership here (a "
                    "mesh-file volume region is known per cell only). Build the region with jno.shape, a "
                    "geometry part or a domain.tag predicate so RCWA can sample it."
                )
            ctx[region_mask_key(v.region)] = jnp.asarray(m, dtype=float)
            continue
        if not isinstance(v, Variable) or v.tag in ctx:
            continue
        if _is_normal_variable(v):  # a source on a flat ambient face reads its (constant) outward normal
            ctx[v.tag] = jnp.broadcast_to(jnp.asarray(_flat_face_normal(v._domain, v.tag[2:])), (pts.shape[0], 3))
        else:
            ctx[v.tag] = pts
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

        from .utils.solver.parametric_helpers import mark_global_scalar

        # one value per solve: a FEM that uses it as a coefficient (an absorbed power as a heat source)
        # reads it once, not per quadrature point
        return mark_global_scalar(FunctionCall(fn, exprs, name=f"rcwa_{method}"), f"__rcwa_{id(self)}_{method}_{args!r}")

    def efficiency(self, kind):
        return self._node("efficiency", kind)

    def order(self, m, n):
        return self._node("order", m, n)

    def power(self, kind="total"):
        return self._node("power", kind)

    def extraction(self, kind="up"):
        return self._node("extraction", kind)

    def aerial(self, NA, source=0.7, defocus=0.0, grid=128, kind="T", polarization=None):
        """The aerial image as a trace node over the mask's ``jno.np.parameter``\\ s -- so ``jax.grad`` of a
        printed-vs-target loss flows back through the imaging AND the RCWA mask solve to the mask design (ILT)."""
        return self._node("aerial", NA, source, defocus, grid, kind, polarization)

    def printed(self, NA, source=0.7, defocus=0.0, grid=128, kind="T", polarization=None, resist=None):
        """The developed resist image as a trace node over the mask's ``jno.np.parameter``\\ s -- ``jax.grad``
        of a printed-vs-target loss flows back through development, imaging AND the RCWA solve to the mask (ILT).
        ``resist`` defaults to :class:`jno.litho.Threshold`; pass another resist model to swap physics."""
        return self._node("printed", NA, source, defocus, grid, kind, polarization, resist or Threshold())

    def expose(self, *a, **k):
        raise RcwaError(
            "expose() returns an optics object, not a scalar readout, so it is not a parametric trace node. "
            "For a differentiable printed-resist loss use rc.solve().printed(resist=...); to inspect an "
            "exposure at concrete values use rc.solve(params={...}).expose(...)."
        )

    def field(self, *a, **k):
        raise RcwaError(
            "field() on a parameterised solve is not a scalar readout; reconstruct it at concrete values "
            "with rc.solve(params={...}).field(...)."
        )


def _profile_solve(run, eng):
    """The profiling path of ``rc.solve(profile=True)`` -- an eager, per-stage-annotated RCWA solve inside a
    JAX Perfetto trace (shared with ``jno.fem``/``jno.fdm``). The eager solve self-blocks (its energy guard
    reads ``efficiency`` concretely), so the shared timer sees the real per-layer state-space transfer work
    (chunked, eig-free)."""
    from jno.utils.profiling import profile_solve

    nt = eng._nt
    label = f"rcwa profile · {len(eng.layers_spec)} layers · n_t={nt} · {2 * nt}×{2 * nt} state transfer/layer"
    return profile_solve(run, label=label)


class _RcwaProblem:
    """An RCWA problem inferred from a jNO constraint list / FEM problem. Holds the inferred
    :class:`RcwaSpec` and builds the state-space engine on :meth:`solve`."""

    def __init__(self, spec, orders, formulation="JONES_DIRECT_FOURIER", resample=None, rpe=None):
        self.spec = spec
        self.orders = orders
        self.formulation = formulation
        self._resample = resample  # params -> ([(thickness, eps_grid), ...], wavelength)  (jax)
        self._rpe = rpe or {}  # {name: parameter-expr} for the trainable parameters in the graph

    def __repr__(self):
        return f"_RcwaProblem(orders={self.orders}, params={sorted(self._rpe)}, {self.spec!r})"

    def _build_engine(self, orders):
        return Rcwa(
            self.spec.layers,
            period=self.spec.period,
            orders=orders,
            wavelength=self.spec.wavelength,
            k_in=self.spec.k_in,
            formulation=self.formulation,
            assume_periodic=True,
            polarization=self.spec.polarization,
        )

    def _engine(self):
        if getattr(self, "_eng", None) is None:  # build ONCE (warmed eagerly in _node) so the expansion is trace-free
            self._eng = self._build_engine(self.orders)
        return self._eng

    def _solve_at(self, params):
        """Concrete solve at explicit parameter values (JAX) -- re-derives eps, wavelength, k_in AND (for an
        internal-source problem) the dipole current."""
        eng = self._engine()
        if not params or self._resample is None:
            return eng.solve(source=self.spec.source)
        layers, wl, kin, source = self._resample(params)
        return eng.solve(layers=layers, wavelength=wl, k_in=kin, source=source)

    def solve(self, params=None, wavelength=None, k_in=None, orders=None, profile=False):
        """Solve.

        The jNO way (**no arguments**): if the constraint list carries ``jno.np.parameter`` coefficients
        (ε, wavelength, ...), ``rc.solve()`` returns a solution whose ``.efficiency(...)`` / ``.order(...)``
        are **trace nodes over those parameters** -- differentiable through crux with no solve args, so
        the exact same parameterised constraint list is differentiable whether you hand it to ``jno.fem``
        or ``jno.rcwa``. A parameter-free problem solves eagerly and returns concrete readouts.

        Escape hatches for a *forward sweep* (not the traced/optimised path): ``params={name: value}`` for
        a concrete solve at explicit values, or ``wavelength`` / ``k_in`` to override those directly.

        ``orders=N`` re-solves at a different Fourier truncation (a fresh engine, the construction ``orders``
        is untouched) -- for a **convergence sweep** (compare ``.efficiency`` at N vs 1.5·N) or profiling at
        scale. ``profile=True`` runs the solve eagerly (at the current parameter values) inside a JAX
        Perfetto trace with per-stage annotations, like ``jno.core.solve(profile=True)`` -- it prints the
        problem size and wall time and writes the trace to ``./rcwa_traces`` (RCWA time is dominated by the
        per-layer state-space transfers -- chunked, eig-free -- which the trace makes explicit)."""
        forward = params is not None or wavelength is not None or k_in is not None or orders is not None or profile
        if not forward:
            if self._rpe and self._resample is not None:
                return _ParametricSol(self, self._rpe)  # parametric -> trace node (crux-differentiable)
            return self._engine().solve(source=self.spec.source)  # parameter-free -> eager concrete
        eng = self._engine() if orders is None else self._build_engine(orders)

        def run():  # a concrete (eager) solve at explicit / current values
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

        return _profile_solve(run, eng) if profile else run()


def _vacuum_ambient_index(first_coeff, last_coeff):
    """Layer index (0 or -1) of the ambient k0 is read from. The coefficient only carries k0^2*eps, so which
    ambient is vacuum is not written in the term list: take the one with the smaller coefficient (a substrate
    under the source has eps = n^2 > 1 and must not rescale k0 to n*k0)."""
    return 0 if first_coeff <= last_coeff else -1


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
    sub=1,
):
    """Build an internal-source (dipole / Gaussian) emission problem: localize the traced ``- f·v`` /
    ``- inner(J, v)`` forcing, split its layer, and route it to the eig-free backend's
    ``jno._rcwa.sources.emission_solve`` (Maxwell jump conditions).

    The Floquet-periodic supercell + absorbing z-ambients behaves as an emitter in a layered environment.
    Scope: scalar Helmholtz layers (no tensor ε̂ / PML yet). The amplitude, orientation, lateral location,
    z-position (within its layer) and Gaussian width, plus the design ε, are all differentiable — only the
    *discrete* choices (which layer, point-vs-Gaussian) are frozen at construction. ``k_in`` is nudged off
    the singular Γ-point."""
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

    # k0 from the vacuum ambient (the smaller coefficient of the two), as in the plane-wave entry (#246)
    ki = _vacuum_ambient_index(*(float(np.real(c[1]).mean()) for c in (cl[0], cl[-1])))
    super_coeff = float(np.real(cl[ki][1]).mean())
    if wavelength is not None:
        k0 = 2 * np.pi / float(wavelength)
    elif super_coeff <= 0:
        raise RcwaError(
            "cannot infer wavelength: the vacuum-ambient permittivity coefficient is non-positive; pass wavelength=."
        )
    else:
        k0 = float(np.sqrt(super_coeff))
    layers = [(t, np.asarray(e) / (k0 * k0)) for t, e in cl]  # scalar relative permittivity

    loc = _localize_source(src_node, is_vector, period, z_range, grid, nz, cparams)
    si = lo = hi = None
    for i in range(1, len(zsp) - 1):  # the source sits in a finite (non-ambient) layer
        zlo, zhi = zsp[i]
        if zlo - 1e-9 <= loc["z0"] <= zhi + 1e-9:
            si, lo, hi = i, zlo, zhi
            break
    if si is None:
        raise RcwaError(
            f"the internal source at z={loc['z0']:.3f} is not inside a finite (non-ambient) layer; place it "
            "within a slab whose permittivity differs from the ambients (a layer contrast defines the slab)."
        )

    # static volume grid for the DIFFERENTIABLE re-derivation of the source's centroid / width from params
    _xs = _cell_centres(period[0], grid)
    _ys = _cell_centres(period[1], grid)
    _zs = np.linspace(z_range[0], z_range[1], nz)
    _GX, _GY, _GZ = np.meshgrid(_xs, _ys, _zs, indexing="ij")
    _pts = np.stack([_GX.ravel(), _GY.ravel(), _GZ.ravel()], 1)

    def geom_at(params):  # differentiable centroid (x0,y0,z0) + lateral σ from the |source|² distribution
        v = _eval_coeff_points(src_node, _pts, params)
        mag2 = jnp.sum(jnp.abs(v) ** 2, -1) if (is_vector and jnp.ndim(v) > 1) else jnp.abs(jnp.reshape(v, (-1,))) ** 2
        w = mag2 / jnp.sum(mag2)
        x0, y0, z0 = (jnp.sum(w * jnp.asarray(_pts[:, k])) for k in range(3))
        sig = 0.5 * (
            jnp.sqrt(jnp.sum((jnp.asarray(_pts[:, 0]) - x0) ** 2 * w))
            + jnp.sqrt(jnp.sum((jnp.asarray(_pts[:, 1]) - y0) ** 2 * w))
        )
        return x0, y0, z0, sig

    def make_source(params):  # amplitude/orientation, location (x0,y0), z-split and fwhm all differentiable
        x0, y0, z0, sig = geom_at(params)
        val = jnp.reshape(_eval_coeff_points(src_node, jnp.stack([x0, y0, z0]).reshape(1, 3), params), (-1,))
        orient = (val[0], val[1], val[2]) if is_vector else (val[0], jnp.asarray(0.0 + 0j), jnp.asarray(0.0 + 0j))
        return dict(
            loc=jnp.stack([x0, y0]).reshape(1, 2),
            layer=si,
            t_upper=hi - z0,  # toward the top ambient (si, lo, hi fixed eagerly; z0 differentiable within the layer)
            t_lower=z0 - lo,
            kind=loc["kind"],
            fwhm=2.35482 * sig,  # FWHM = 2.355·σ (used only when kind == "gaussian")
            orient=orient,
            amp=1.0,
        )

    kin = (1e-3, 1e-3)  # nudge off the singular Γ-point (exactly-normal k_in is degenerate for an internal source)

    def resample(params):
        g = grid * sub  # supersampled lattice for subpixel smoothing
        k0p = jnp.sqrt(jnp.mean(jnp.real(_eval_coeff_points(coeff_node, _cell_grid_at_z(period, g, zm[ki]), params))))
        out = []
        for (thick, _), zmid in zip(layers, zm):
            cfine = jnp.reshape(_eval_coeff_points(coeff_node, _cell_grid_at_z(period, g, zmid), params), (g, g))
            out.append((thick, _pixel_average(cfine, grid, sub) / (k0p * k0p)))  # keep complex ε (absorption)
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


def _source_jones(src_coeff, period, z0, params):
    """The incident transverse Jones vector ``(E_x, E_y)`` of a VECTOR source (``inner(cross(E_inc, n), v×n)``), or
    None for a scalar one. Read at the face centre; the plane-wave phase is common to every component."""
    pt = np.array([[period[0] * 0.5, period[1] * 0.5, z0]])
    v = np.asarray(_eval_coeff_points(src_coeff, pt, params)).reshape(-1)
    if v.size < 2:
        return None
    if abs(v[0]) ** 2 + abs(v[1]) ** 2 < 1e-24 * max(1.0, float(np.max(np.abs(v)) ** 2)):
        raise RcwaError(
            "the vector incident wave has no transverse (x, y) component at the illuminated face, so it "
            "carries no power into the stack; give E_inc a component in the plane of the face."
        )
    return _as_jones((complex(v[0]), complex(v[1])))


def _scalar_polarization(layers, k_in):
    """The polarization a SCALAR Helmholtz list stands for, and why.

    Maxwell reduces EXACTLY to the scalar equation ``Δu + k0²ε u = 0`` for the field component along an
    in-plane axis ``a`` when every layer is invariant along ``a`` and the incidence has no component along
    ``a`` (E ∥ a is then TE -- s-polarized): a 1-D grating in the classical mount, or any uniform stack.
    That is the polarization solved, so ``jno.rcwa`` and ``jno.fem`` answer the same equation. A stack
    patterned along both axes (or illuminated off its invariant axis) has no such polarization; it is then
    solved x-polarized, and the returned note says so. Returns ``(jones, note)``."""
    if any(isinstance(e, tuple) for _, e in layers):
        return _as_jones("x"), (
            "x by default: an anisotropic or PML stack reduces to the scalar equation for no polarization; "
            "pass polarization= to choose, or write the vector (curl-curl) list"
        )
    vary_x = vary_y = False
    for _, e in layers:
        g = np.asarray(e)
        if g.ndim != 2:
            continue
        tol = 1e-9 * max(1.0, float(np.max(np.abs(g))))
        vary_x |= bool(np.max(np.abs(g - g[:1, :])) > tol)  # changes along x (axis 0)
        vary_y |= bool(np.max(np.abs(g - g[:, :1])) > tol)  # changes along y (axis 1)
    kx, ky = (float(k) for k in k_in)
    kt = 1e-9 * max(1.0, abs(kx), abs(ky))
    if not vary_y and abs(ky) <= kt and (vary_x or abs(kx) > kt):
        return _as_jones(
            "y"
        ), "y: every layer is invariant along y and the incidence has no y component, so E_y is the scalar field (TE)"
    if not vary_x and abs(kx) <= kt:
        why = (
            "the stack is uniform"
            if not vary_y
            else "every layer is invariant along x and the incidence has no x component"
        )
        return _as_jones("x"), f"x: {why}, so E_x is the scalar field (TE)"
    if not vary_x and not vary_y:  # a uniform stack at any azimuth: s-polarized, E = z x k_in
        return _as_jones(
            (-ky, kx)
        ), "s: the stack is uniform, so the field perpendicular to the plane of incidence is the scalar field"
    return _as_jones("x"), (
        "x by default: the stack is patterned along both in-plane axes (or lit off its invariant axis), so "
        "Maxwell reduces to the scalar equation for no polarization; pass polarization= to choose, or write "
        "the vector (curl-curl) list"
    )


def rcwa(
    problem,
    *,
    orders,
    wavelength=None,
    grid=64,
    nz=64,
    slices=None,
    params=None,
    formulation="JONES_DIRECT_FOURIER",
    smoothing=1,
    polarization=None,
):
    """Infer and build an RCWA problem from a jNO constraint list (or built ``FEM``).

    Everything is read out of the traced problem: **periodicity + period** (Floquet ties — absent ⇒
    raise), the **super/substrate ambients**, the **permittivity** (the ``K0**2*eps`` coefficient
    recovered from the scalar Helmholtz volume term, sampled along z; a tensor ``ε̂`` from
    ``inner(ε̂ @ u, v)``), the **wavelength** (``k0`` from the vacuum ambient, the one with the smaller coefficient, unless ``wavelength``
    is given), and the **incident wave** (illuminated face + transverse angle ``k_in``, from the
    assembled forcing). An **in-plane PML** (a complex coordinate stretch in the stiffness coefficients)
    is detected and honoured as a diagonal Maxwell ``ε̂``/``μ̂`` — the supercell then behaves as an
    isolated scatterer, and a design parameter on the scatterer ε **is differentiable** (inverse design of
    an isolated structure). Scope: uniaxial (diagonal) in-plane stretch only, scalar Helmholtz.

    An **internal source** (a ``- f·v`` / ``- inner(J, v)`` volume forcing) switches the solve to an
    emission problem: the source is localized (point vs Gaussian; dipole orientation from ``J``), the stack
    is split at the source plane, and ``sol.power("up"|"down"|"total")`` / ``sol.extraction("up"|"down")``
    give the radiated power and directionality. The amplitude, orientation, lateral location, z-position and
    Gaussian width, plus the design ε, are all differentiable (only the discrete layer choice is frozen);
    ``k_in`` is nudged off the singular Γ-point, and the source must sit in a finite (contrast-defined) layer.
    Scalar Helmholtz only for now; a boundary incidence + internal source together raise.

    Args:
        problem: The same constraint list you would pass to :func:`jno.fem`, or an already-built
            ``FEM``.
        orders: Fourier truncation for the state-space solve.
        wavelength: Free-space wavelength (same length unit as the geometry). Optional — inferred from
            the superstrate (assumed vacuum) when omitted. Pass it to override or if no ambient is
            vacuum.
        grid: Transverse resolution used to detect the layer stack.
        nz: Number of z-samples used to detect the layer stack.
        slices: Staircase a continuously-varying permittivity into this many layers instead of raising.
        params: values of the problem's ``jno.np.parameter`` nodes, by name, used for the eager
            inference (structure, source, wavelength) instead of their initial values.
        formulation: the Fourier-factorization name, e.g. ``"JONES_DIRECT_FOURIER"`` (default; converges
            well for high-contrast dielectrics, where the naive ``"FFT"`` rule converges poorly). The
            state-space backend implements all eight fmmax vector schemes plus ``FFT`` and a callable
            tangent-field hook.
        smoothing: Subpixel-smoothing factor (default ``1`` = off). ``smoothing=k`` supersamples each
            RCWA pixel ``k×k`` and area-averages the permittivity, anti-aliasing material boundaries.
            Recommended (``2``–``4``) for **inverse design**: it cuts Fourier (Gibbs) ringing so fewer
            ``orders`` converge, and makes the gradient w.r.t. a boundary-moving design parameter smooth
            rather than staircased. Costs ``k²``× more coefficient evaluations (cheap) — the
            state-space solve size is unchanged.
        polarization: The incident polarization: ``"x"``, ``"y"``, or a transverse Jones vector ``(p_x,
            p_y)`` (complex entries give elliptical light). Usually inferred, and the choice is logged
            and kept on ``spec.polarization`` / ``spec.polarization_from``. A **vector** (curl-curl)
            list carries it in its incident field ``E_inc``. A **scalar** Helmholtz list is solved in
            the polarization for which Maxwell reduces exactly to the written equation -- E along the
            axis every layer is invariant along, when the incidence has no component along it (a 1-D
            grating: E parallel to its lines; TE) -- so ``jno.rcwa`` and ``jno.fem`` answer the same
            equation. A stack patterned along both axes has no such polarization; it is solved
            x-polarized, and the log says so.

    Returns:
        _RcwaProblem: Holds the inferred :class:`RcwaSpec` (``.spec``) and a
            :meth:`~_RcwaProblem.solve`.
    """
    sub = int(smoothing)
    if sub < 1:
        raise RcwaError(f"smoothing must be a positive integer (1 = off), got {smoothing!r}.")
    femobj = _as_fem(problem)
    domain = femobj.domain
    from .geometry.parametric import refuse_if_parametric

    refuse_if_parametric(domain, "jno.rcwa", "its layers are detected from the reference mesh as numbers")
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
        C, zs = _sample_grid(domain, coeff_nodes, grid, nz, period, z_range, sub=sub)
        locate = None if slices is not None else _interface_bisection_nodal(pts, coeff_nodes, grid, period, sub)
        if slices is None:
            from jno.utils.logger import get_logger

            get_logger().info(
                "rcwa: the permittivity is a mesh field, so each layer interface is placed where its "
                "interpolant crosses halfway between the layers -- as sharp as the mesh, no sharper."
            )
    else:  # analytic permittivity -> sample the grid exactly
        C, zs = _sample_grid_direct(coeff_node, grid, nz, period, z_range, cparams, sub=sub)
        locate = None if slices is not None else _interface_bisection(coeff_node, grid, period, cparams, sub)
    coeff_layers = detect_layers(C, zs, slices=slices, locate=locate)
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
            sub,
        )

    face_z, k_in = _source_kin(femobj, domain, cparams)
    source_at_bottom = abs(face_z - _region_centroid(domain, bottom)[2]) < abs(face_z - _region_centroid(domain, top)[2])
    source_face = bottom if source_at_bottom else top

    # orient so the incident (source-side) ambient is layers[0] = superstrate
    if not source_at_bottom:
        coeff_layers = list(reversed(coeff_layers))
        zmids = list(reversed(zmids))

    # k0 from the VACUUM ambient (coeff = k0^2), unless wavelength is given. The coefficient only carries
    # k0^2*eps, so which ambient is vacuum is not written in the term list: take the one with the smaller
    # coefficient (a substrate under the source has eps = n^2 > 1 and must not rescale k0 to n*k0). With a PML
    # the ambient is a vacuum framed by the absorber, so its transverse MEAN is stretched; read k0 at the
    # physical-core centre instead (there Λ=1 and ε=1, so the mass coefficient is exactly k0²).
    def _ambient_coeff(i):
        if is_pml:
            center = np.array([[period[0] * 0.5, period[1] * 0.5, zmids[i]]])
            return float(np.real(_eval_coeff_points(coeff_node, center, cparams)).reshape(-1)[0])
        return float(np.real(coeff_layers[i][1]).mean())

    ki = _vacuum_ambient_index(_ambient_coeff(0), _ambient_coeff(-1))  # the ambient k0 is read from (layer index)
    super_coeff = _ambient_coeff(ki)
    if wavelength is not None:
        k0 = 2 * np.pi / float(wavelength)
    elif super_coeff <= 0:
        raise RcwaError(
            "cannot infer wavelength: the vacuum-ambient permittivity coefficient is non-positive; pass wavelength=."
        )
    else:
        k0 = float(np.sqrt(super_coeff))
    k0sq = k0 * k0
    if is_pml:  # in-plane uniaxial PML: general-anisotropic layer (ε̂ AND μ̂), 10 grids per layer
        layers = [
            (
                t,
                tuple(
                    np.asarray(c) for c in _pml_layer_components(coeff_node, stretch, period, grid, zm, cparams, k0sq, sub)
                ),
            )
            for (t, _), zm in zip(coeff_layers, zmids)
        ]
    elif is_aniso:  # relative-permittivity TENSOR per layer: (ε_xx, ε_xy, ε_yx, ε_yy, ε_zz)
        layers = [
            (t, tuple(np.asarray(c) for c in _tensor_components(coeff_node, period, grid, zm, cparams, k0sq, sub)))
            for (t, _), zm in zip(coeff_layers, zmids)
        ]
    else:
        layers = [(t, np.asarray(e) / k0sq) for t, e in coeff_layers]  # relative permittivity (scalar)

    # differentiable re-sampling: re-evaluate the permittivity coefficient at each layer's representative z
    # from a parameter dict, so a design (jno.np.parameter) flows through the solve. Re-deriving the WHOLE
    # coefficient K0^2*eps handles an EPS/shape parameter (each layer's eps) AND a WAVELENGTH parameter
    # (k0 = sqrt(coeff in the vacuum ambient)) in one shot.
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
    if nodal_names:  # bary at grid·sub -> the nodal density is interpolated on the supersampled lattice, then averaged
        nc = np.asarray(domain.points)
        bary = [_bary_weights(nc, _cell_grid_at_z(period, grid * sub, zm)) for zm in zmids]  # per layer: (ids, weights)

    # the illuminated boundary term's forcing -> lets an incidence-ANGLE parameter in the source flow
    # (its transverse phase gradient IS k_in); None when the source has no angle to read.
    src_coeff = next(
        (sc for c in getattr(femobj, "_constraints", []) if (sc := _extract_source_coeff(c)) is not None), None
    )
    src_z = float(_region_centroid(domain, source_face)[2])
    if src_coeff is not None:  # the source field's phase is the robust k_in (a b-based read is edge-sign-
        k_in = tuple(float(x) for x in _kin_from_source(src_coeff, period, src_z, cparams))  # corrupted for N1E)

    def resample(params):
        g = grid * sub  # supersampled lattice for subpixel smoothing

        def at(li):  # parameter values for layer li: nodal fields barycentrically interpolated, scalars as-is
            def gather(v):
                ids, w = bary[li]
                return jnp.sum(jnp.asarray(w) * jnp.asarray(v).reshape(-1)[jnp.asarray(ids)], axis=1)

            return {k: (gather(v) if k in nodal_names else jnp.asarray(v)) for k, v in params.items()}

        if is_pml:  # in-plane PML: k0 at the physical-core centre (Λ=1, ε=1 ⇒ mass = k0²); 10 ε̂/μ̂ grids per layer
            center_pt = np.array([[period[0] * 0.5, period[1] * 0.5, zmids[ki]]])
            k0 = jnp.sqrt(jnp.real(jnp.reshape(_eval_coeff_points(coeff_node, center_pt, at(ki)), (-1,))[0]))
            out = [
                (thick, _pml_layer_components(coeff_node, stretch, period, grid, zmid, at(li), k0 * k0, sub))
                for li, (thick, zmid) in enumerate(zip(thicks, zmids))
            ]
        elif is_aniso:  # k0 from the superstrate's xx (isotropic vacuum ⇒ xx = k0²); tensor components per layer
            sup_xx = jnp.reshape(
                _eval_coeff_points(coeff_node, _cell_grid_at_z(period, g, zmids[ki]), at(ki)), (g, g, 3, 3)
            )
            k0 = jnp.sqrt(jnp.mean(jnp.real(sup_xx[..., 0, 0])))
            out = [
                (thick, _tensor_components(coeff_node, period, grid, zmid, at(li), k0 * k0, sub))
                for li, (thick, zmid) in enumerate(zip(thicks, zmids))
            ]
        else:
            sup = jnp.mean(jnp.real(_eval_coeff_points(coeff_node, _cell_grid_at_z(period, g, zmids[ki]), at(ki))))
            k0 = jnp.sqrt(sup)
            out = []
            for li, (thick, zmid) in enumerate(zip(thicks, zmids)):
                cfine = jnp.reshape(_eval_coeff_points(coeff_node, _cell_grid_at_z(period, g, zmid), at(li)), (g, g))
                # keep the FULL complex coefficient: ε = coeff / k0² is complex for a lossy layer; taking the
                # real part here would silently drop the absorption (Im ε) — k0 stays real (vacuum superstrate).
                out.append((thick, _pixel_average(cfine, grid, sub) / (k0 * k0)))
        kin = _kin_from_source(src_coeff, period, src_z, params) if src_coeff is not None else jnp.asarray(k_in)
        return out, 2 * jnp.pi / k0, kin, None  # source=None: plane-wave incidence, not an internal source

    # the trainable jno.np.parameter(s) the permittivity depends on -> solve() returns trace nodes over
    # them (crux threads the values), so the parameterised constraint list is differentiable with no solve
    # args -- the same way jno.fem would differentiate it. (Collected from the bare coefficient tree; the
    # ScalarView-wrapped raw constraints aren't traversed by the collector.)
    from jno.utils.solver.parametric_helpers import _collect_runtime_parameter_exprs

    rpe = {}
    _collect_runtime_parameter_exprs(coeff_node, rpe)  # eps / wavelength parameters (PML: the scatterer ε too)
    if src_coeff is not None:
        _collect_runtime_parameter_exprs(src_coeff, rpe)  # an incidence-angle parameter in the source

    if polarization is not None:
        pol, pol_from = _as_jones(polarization), "given"
    else:
        vpol = _source_jones(src_coeff, period, src_z, cparams) if src_coeff is not None else None
        if vpol is not None:
            pol, pol_from = vpol, "read from the vector incident field"
        else:
            pol, pol_from = _scalar_polarization(layers, k_in)
    from jno.utils.logger import get_logger

    get_logger().info(f"rcwa: incident polarization (px, py) = ({pol[0]:.3g}, {pol[1]:.3g}) -- {pol_from}")

    spec = RcwaSpec(
        period=period,
        layers=layers,
        wavelength=2 * np.pi / k0,
        k_in=k_in,
        source_face=source_face,
        ambient_faces=(bottom, top),
        periodic_axes=axes,
        polarization=pol,
        polarization_from=pol_from,
    )
    return _RcwaProblem(spec, orders=orders, formulation=formulation, resample=resample, rpe=rpe)
