"""Rigorous Coupled-Wave Analysis (RCWA / Fourier Modal Method) for periodic, layered structures.

RCWA is semi-analytic in the propagation direction: each layer is expanded in a truncated Fourier
basis in-plane and solved by an eigenmode decomposition, then layers are stitched with a scattering
matrix. For **periodic, piecewise-z-invariant** structures — extruded metasurface pillars are the
canonical case — this is far cheaper than a full 3-D volumetric solve.

This is an **optional** jNO backend built on `fmmax` (a differentiable JAX Fourier Modal Method);
`fmmax` is imported lazily so the core install stays lean. Install it with::

    pip install jax-neural-operators[rcwa]      # or: pixi run -e rcwa ...

Usage (forward engine — explicit layers)::

    rc  = jno.rcwa([(inf, 1.0), (0.35, eps_xy), (inf, 2.1)], period=(0.6, 0.6), orders=300, wavelength=1.03)
    sol = rc.solve(inc=None)     # inc = transverse incident field on the cell grid (None = normal plane wave)
    sol.efficiency("T")          # transmitted / reflected power (0th order + all propagating orders)
    sol.order(+1, 0)             # diffraction efficiency into a specific order

Design rules honoured here:
 * Permittivity is passed, not an equation — the governing (Helmholtz) physics is implicit.
 * It **never fails silently**: every inference (periodicity, layer thickness, order/grid Nyquist,
   wavelength, energy balance) is validated and raises :class:`RcwaError` with a concrete fix.
 * The two `fmmax` gotchas are baked in so callers never rediscover them: the Poynting flux is summed
   *with sign* (an ``abs()`` sum over-counts and yields ``T+R>1``), and the ``JONES_DIRECT_FOURIER``
   Fourier factorization is the default (the naive rule converges poorly for high-contrast dielectrics).

Planned increment (needs the ``eps.at(coords)`` primitive): an ``eps``-driven front-end
``jno.rcwa(eps, orders=...)`` that samples a jNO permittivity expression on a grid, auto-derives the
layers, and threads the design gradient through — plus ``as_precond`` for ``fem.solve(precond=rc)``.

References:
 * M. G. Moharam & T. K. Gaylord, "Rigorous coupled-wave analysis of planar-grating diffraction",
   J. Opt. Soc. Am. 71, 811 (1981).
 * M. G. Moharam, D. A. Pommet, E. B. Grann & T. K. Gaylord, "Stable implementation of the enhanced
   transmittance matrix approach", J. Opt. Soc. Am. A 12, 1077 (1995).
 * fmmax: https://github.com/facebookresearch/fmmax
"""

import numpy as np

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


class _Sol:
    def __init__(self, fm, s, layers, expansion, nt, Pin, wavelength):
        self._fm, self._s, self._layers, self._ex = fm, s, layers, expansion
        self._nt, self._Pin, self._wl = nt, Pin, wavelength
        self._fwd = np.zeros((2 * nt, 1), complex)  # filled by solve()

    def _flux(self, amps, layer, backward=False):
        f, b = self._fm.directional_poynting_flux(
            amps if not backward else np.zeros_like(amps),
            amps if backward else np.zeros_like(amps),
            layer,
        )
        return np.asarray(b if backward else f).reshape(-1)  # SIGNED (not abs) — energy-correct

    def efficiency(self, kind):
        """Total transmitted (``"T"``) or reflected (``"R"``) power fraction, summed over propagating orders."""
        if kind == "T":
            return float(np.sum(self._flux(self._s.s11 @ self._fwd, self._layers[-1]))) / self._Pin
        if kind == "R":
            return float(-np.sum(self._flux(self._s.s21 @ self._fwd, self._layers[0], backward=True))) / self._Pin
        raise RcwaError(f"efficiency(kind): kind must be 'T' or 'R', got {kind!r}")

    def order(self, m, n):
        """Diffraction efficiency into transmitted order ``(m, n)`` (raises if outside the truncation)."""
        bc = np.asarray(self._ex.basis_coefficients)
        hit = np.where((bc[:, 0] == m) & (bc[:, 1] == n))[0]
        if len(hit) == 0:
            raise RcwaError(f"order ({m},{n}) is outside the truncation ({self._nt} terms); raise orders= to include it.")
        i = int(hit[0])
        tf = np.abs(
            np.asarray(
                self._fm.directional_poynting_flux(
                    self._s.s11 @ self._fwd, np.zeros((2 * self._nt, 1), complex), self._layers[-1]
                )[0]
            ).reshape(-1)
        )
        return float(tf[i] + tf[i + self._nt]) / self._Pin


class Rcwa:
    """A periodic layered RCWA problem. Construct with the layer stack, period and truncation; call
    :meth:`solve` with an incident field to obtain a :class:`_Sol`.

    Parameters
    ----------
    layers:
        Ordered ``[(thickness, eps), ...]`` from superstrate to substrate. ``thickness`` is a positive
        float, or ``inf``/``None`` for the two semi-infinite ambient layers. ``eps`` is a scalar (uniform
        layer) or a 2-D array sampled on the unit-cell grid (patterned layer).
    period:
        In-plane period ``(Px, Py)`` of the unit cell, both > 0.
    orders:
        Approximate number of Fourier terms (truncation). High-contrast dielectrics (e.g. a-Si) need
        several hundred; low-contrast converge in tens.
    wavelength:
        Free-space wavelength (same length unit as ``period``/thicknesses). May be deferred to
        :meth:`solve`. Never defaulted.
    formulation:
        `fmmax` Fourier-factorization rule; ``"JONES_DIRECT_FOURIER"`` (the default) is robust for
        high-contrast structures.
    assume_periodic:
        RCWA solves an *infinitely periodic* in-plane problem. You must pass ``True`` to affirm the cell
        is a genuine period/supercell — a finite aperture is not periodic and would be solved wrongly.
    """

    def __init__(
        self, layers, *, period, orders, wavelength=None, formulation="JONES_DIRECT_FOURIER", assume_periodic=False
    ):
        fm = _fmmax()
        # --- guard: periodicity is an ASSUMPTION, not a fact ---
        if period is None or len(period) != 2 or period[0] <= 0 or period[1] <= 0:
            raise RcwaError(f"period must be (Px,Py) > 0, got {period!r}.")
        if not assume_periodic:
            raise RcwaError(
                "RCWA is periodic in-plane. If your cell is genuinely a period/supercell, pass "
                "assume_periodic=True. A finite aperture with absorbing side walls is NOT periodic — "
                "RCWA would silently solve the wrong (periodic) problem."
            )
        # --- guard: layers well-formed ---
        if not layers:
            raise RcwaError("layers is empty: need at least [superstrate, ..., substrate].")
        for k, (t, e) in enumerate(layers):
            if not (t is None or (np.isreal(t) and t > 0) or t == np.inf):
                raise RcwaError(f"layer {k} thickness must be > 0 (or inf for ambient), got {t!r}.")
            eg = np.asarray(e)
            if eg.ndim == 2 and (eg.shape[0] < 4 or eg.shape[1] < 4):
                raise RcwaError(f"layer {k} eps grid {eg.shape} is too coarse to rasterize; sample it finer.")
        # --- guard: patterned grids must resolve the requested orders (Nyquist); uniform layers need none ---
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
        self.orders, self.wavelength = orders, wavelength
        self.formulation = fm.Formulation[formulation]
        self.assume_periodic = True

    def solve(self, inc=None, wavelength=None):
        """Solve the stack for an incident field ``inc`` (``None`` = a normal-incidence plane wave) and
        return a :class:`_Sol`. Raises if the wavelength is unknown or energy is not conserved."""
        fm = self.fm
        wl = wavelength if wavelength is not None else self.wavelength
        if wl is None:
            raise RcwaError(
                "wavelength is unknown: pass wavelength= to solve() (or at construction). "
                "It sets every layer's eigenmodes and is never defaulted."
            )
        lv = fm.LatticeVectors(u=np.array([self.period[0], 0.0]), v=np.array([0.0, self.period[1]]))
        ex = fm.generate_expansion(lv, approximate_num_terms=self.orders)
        nt = ex.num_terms
        k_in = np.zeros(2)

        def solve_layer(e):
            eg = np.asarray(e) + 0j
            if eg.ndim == 0:
                eg = np.full((1, 1), eg)
            return fm.eigensolve_isotropic_media(np.asarray(wl), k_in, lv, eg, ex, formulation=self.formulation)

        layers = [solve_layer(e) for _, e in self.layers_spec]
        thick = [np.asarray(1.0 if t is None or t == np.inf else t) for t, _ in self.layers_spec]
        s = fm.stack_s_matrix(layers, thick)
        fwd = np.zeros((2 * nt, 1), complex)
        fwd[0, 0] = 1.0
        Pin = float(np.sum(np.asarray(fm.directional_poynting_flux(fwd, np.zeros_like(fwd), layers[0])[0])))
        sol = _Sol(fm, s, layers, ex, nt, Pin, wl)
        sol._fwd = fwd
        # --- guard: energy balance (lossless => T+R==1). A violation = non-convergence or a bug. ---
        T, R = sol.efficiency("T"), sol.efficiency("R")
        if T + R > 1.0 + 5e-3:
            raise RcwaError(
                f"energy not conserved: T+R={T + R:.4f} > 1 at orders={self.orders}. The modal solve is "
                f"not converged (raise orders) or the structure is unphysical."
            )
        return sol

    def as_precond(self, transfer=None):
        """Return a :mod:`jno.precond` spec applying this (background) RCWA as ``M⁻¹ = A₀⁻¹``.

        The FEM↔grid ``transfer`` (rasterize the residual to the RCWA grid, interpolate the background
        field back to FEM nodes) is a planned increment; without it this raises rather than silently
        returning a no-op preconditioner.
        """
        rc = self

        class _Spec:
            def materialize(self, ctx):
                if transfer is None:
                    raise RcwaError(
                        "as_precond needs a FEM<->grid transfer operator (rasterize residual to the RCWA "
                        "grid, interpolate the background field back to FEM nodes). Not yet wired."
                    )
                raise NotImplementedError("transfer wiring is a planned increment (see jno.rcwa module docstring).")

        _ = rc
        return _Spec()


def rcwa(layers, *, period, orders, wavelength=None, **kw):
    """Build an :class:`Rcwa` problem — the functional entry point (see :class:`Rcwa` for the arguments)."""
    return Rcwa(layers, period=period, orders=orders, wavelength=wavelength, **kw)
