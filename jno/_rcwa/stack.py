"""Chunked R-matrix/impedance recursion (Li 1996) and the stack solve.

The stack is solved entirely in state space -- no directional decomposition
anywhere but the two analytic uniform ambients:

* Each layer is split into same-material sub-slabs (chunks) at BUILD time; the
  chunk count is a static integer from the concrete spectral bound
  ``|q|_max * d`` (``q`` = modal wavenumber, bounded by
  ``sqrt(|omega^2 eps|_max + max|k+G|^2)``). Same-material chunking is exact
  (``expm(i M d) == prod_k expm(i M d_c)``); an un-chunked thick evanescent
  layer breaks *by design* -- that is the stress the recursion exists for.
* Chunks are combined by the impedance (R-matrix) recursion of Li 1996: with
  the half-space below an interface represented by its surface admittance
  ``Y`` (``v = Y u``), a chunk with transfer ``T = [[A, B], [C, D]]`` (mapping
  top -> bottom) pushes the admittance upward:

      Y_top = (D - Y_bottom B)^{-1} (Y_bottom A - C).

  All quantities stay O(1); growing exponentials never appear as matrix
  entries -- they are absorbed by the transfer combined with the admittance.
* The only analytic plane-wave splits are at the two ambients: per-order
  ``kz = sqrt(eps*omega^2 - |k+G|^2)`` with the ``Im(kz) >= 0`` branch, the
  per-order surface admittance, and the backward-wave sign flip
  ``v_ref = -Y_sup u_ref``. Then

      u_ref = (Y_stack + Y_sup)^{-1} (Y_sup - Y_stack) u_inc ,

  and the transmitted state is obtained by propagating the STATE VECTOR chunk
  by chunk (never forming the full 4n transfer product).

Readouts follow jNO's conventions exactly: signed per-order Poynting
``S_z = 0.5 Re(u1 conj(v1) + u2 conj(v2))`` summed over orders, ``T`` from the
transmitted state, ``R`` from the reflected state with the sign flip,
normalized by the incident power.

Jit hygiene: chunk counts, Pade squaring counts, uniform/patterned flags and
thicknesses are static Python values decided at build from the concrete
prototype grids; the grids themselves flow through pure JAX inside the traced
core (native JVPs). There are no data-dependent loops in the compiled program.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from . import basis as basis_mod
from . import layer, matrices
from .matrices import Formulation

#: Per-chunk growth measure kept <= this by the build-time chunking.
_CHUNK_THETA = 1.0
#: House energy guard, exactly jNO's rcwa convention.
_ENERGY_TOL = 5e-3

_DEFAULT_FORMULATION = Formulation.JONES_DIRECT_FOURIER


def _concrete(x) -> bool:
    return not isinstance(x, jax.core.Tracer)


def _energy_guard(res, auto_chunks):
    """The never-silent energy guard (house style): rejects NaN/inf efficiencies,
    ``T + R > 1 + tol`` (lossless stacks) and negative efficiencies with an
    actionable message. Eager host-side only -- callers gate on ``_concrete``.

    ``auto_chunks`` is the build-time per-layer chunk count, reported so the
    T+R>1 cliff (the un-chunked impedance recursion) points at its remedy.
    """
    if not (math.isfinite(float(res.T)) and math.isfinite(float(res.R))):
        raise ValueError(
            f"_rcwa: non-finite efficiency T={float(res.T):.4f} R={float(res.R):.4f} -- "
            "the transfer overflowed; chunk the layer thinner (more sub-slabs) or raise orders."
        )
    if float(res.T + res.R) > 1.0 + _ENERGY_TOL:
        raise ValueError(
            f"_rcwa: energy guard violated T+R={float(res.T + res.R):.4f} > 1 + "
            f"{_ENERGY_TOL} (lossless stack) -- the solve is not converged: raise orders, "
            f"or chunk thick high-contrast layers thinner (the T+R>1 cliff is the "
            f"un-chunked recursion; build-time auto chunks were {auto_chunks})."
        )
    if float(res.T) < 0.0 or float(res.R) < 0.0:
        raise ValueError(
            f"_rcwa: negative efficiency T={float(res.T):.4f} R={float(res.R):.4f} -- "
            "nonphysical (transfer overflowed); chunk the layer thinner or raise orders."
        )


# -----------------------------------------------------------------------------
# Counting oracle (the seam's analogue of fmmax's patterned-eigensolve counter).
# -----------------------------------------------------------------------------

#: How many patterned-layer transfers were computed since the last reset. This
#: counts *traces* of the compiled core (Python executes during tracing, not on
#: a jitted cache hit), which is exactly the ``test_a_repeat_solve_runs_the_
#: compiled_core`` semantics the fmmax monkeypatch used to provide.
_PATTERNED_SOLVES = [0]


def patterned_solve_count() -> int:
    """The number of patterned-layer expm solves since the last reset."""
    return _PATTERNED_SOLVES[0]


def reset_patterned_solve_count() -> None:
    """Zero the patterned-solve counter."""
    _PATTERNED_SOLVES[0] = 0


def count_patterned_solves(fn):
    """Run ``fn`` with the patterned-layer solve counted; return ``(result, n_calls)``.

    ``n_calls`` counts traced executions of the compiled core, one per
    patterned layer per trace -- the stage-1 oracle behind
    ``test_uniform_grid_layers_take_the_analytic_eigensolve`` and the repeat-
    solve jit-reuse tests (fmmax's ``_eigensolve_patterned_isotropic_media``
    monkeypatch, replaced).
    """
    reset_patterned_solve_count()
    out = fn()
    return out, patterned_solve_count()


def spectral_bound(eps_grid, twv, omega) -> float:
    """Static bound on ``max|q| = sqrt(|omega^2 eps| + |k+G|^2)`` (host-side)."""
    import numpy as onp

    eps = onp.asarray(eps_grid)
    max_eps = float(onp.max(onp.abs(eps)))
    max_kt2 = float(onp.max(onp.real(onp.asarray(twv[..., 0]) ** 2 + onp.asarray(twv[..., 1]) ** 2)))
    om = float(onp.max(onp.abs(onp.asarray(omega))))
    return math.sqrt(max_eps * om * om + max_kt2)


def chunk_count(thickness: float, rho: float, theta: float = _CHUNK_THETA) -> int:
    """Static same-material sub-slab count: ``ceil(rho * d / theta)``, >= 1."""
    return max(1, int(math.ceil(rho * thickness / theta)))


def _polarization_vector(polarization) -> Tuple[complex, complex]:
    """Jones tuple from ``"x"``/``"y"``/pair input (eager, concrete)."""
    if isinstance(polarization, str):
        if polarization == "x":
            return (1.0, 0.0)
        if polarization == "y":
            return (0.0, 1.0)
        raise ValueError(f"polarization must be 'x', 'y', or (px, py); got {polarization!r}.")
    px, py = complex(polarization[0]), complex(polarization[1])
    if abs(px) ** 2 + abs(py) ** 2 <= 0:
        raise ValueError(f"polarization must be nonzero, got {(px, py)}.")
    return (px, py)


class StateStack:
    """Eig-free state-space RCWA stack with the jNO seam shape.

    Args:
        wavelength: Free-space wavelength (the build-time value; may be
            overridden per :meth:`solve`).
        period: Lattice periods ``(Px, Py)``.
        amb_sup, amb_sub: Superstrate and substrate permittivities.
        layers: The stack ``[(thickness, eps), ...]`` from top to bottom;
            `eps` is a scalar or a permittivity grid.
        orders: Approximate number of Fourier orders (jNO's ``orders``).
        k_in: Incident transverse wavevector ``(kx, ky)`` (fixed per stack).
        formulation: ``Formulation`` (or its name) for patterned-layer Fourier
            factorization; defaults to ``JONES_DIRECT_FOURIER`` like ``jno.rcwa``.
        theta: Per-chunk growth bound for the static chunking policy.
        chunk_override: Optional per-layer chunk counts (tests / stress).
        squarings_override: Optional fixed Pade squaring count for all
            patterned layers (tests / stress).
        dtype: The complex dtype of the solve (``complex64`` default).
    """

    def __init__(
        self,
        wavelength: float,
        period: Tuple[float, float],
        amb_sup: float,
        amb_sub: float,
        layers: Sequence[Tuple[float, jnp.ndarray]],
        orders: int = 200,
        k_in: Tuple[float, float] = (0.0, 0.0),
        formulation: Formulation | str = _DEFAULT_FORMULATION,
        theta: float = _CHUNK_THETA,
        chunk_override: Optional[Sequence[Optional[int]]] = None,
        squarings_override: Optional[int] = None,
        dtype=None,
    ):
        if dtype is None:
            dtype = jnp.complex64
        if isinstance(formulation, str):
            try:
                formulation = Formulation[formulation]
            except KeyError:
                raise ValueError(
                    f"unknown formulation {formulation!r}; valid names: {', '.join(m for m in Formulation.__members__)}"
                ) from None
        if not isinstance(formulation, Formulation) and not callable(formulation):
            raise TypeError(f"`formulation` must be a Formulation (or name) or a callable, got {formulation!r}.")
        if wavelength is None or not (math.isfinite(float(wavelength)) and wavelength > 0):
            raise ValueError(f"`wavelength` must be a positive finite value, got {wavelength!r}.")
        if len(period) != 2 or period[0] <= 0 or period[1] <= 0:
            raise ValueError(f"`period` must be (Px, Py) > 0, got {period!r}.")
        self._cdt = dtype
        self.wavelength = float(wavelength)
        self.period = tuple(float(p) for p in period)
        self.amb_sup, self.amb_sub = complex(amb_sup), complex(amb_sub)
        self.k_in = tuple(float(x) for x in k_in)
        self.formulation = formulation
        self.layers = list(layers)

        lv = basis_mod.LatticeVectors(
            u=jnp.asarray([self.period[0], 0.0]),
            v=jnp.asarray([0.0, self.period[1]]),
        )
        expansion = basis_mod.generate_expansion(lv, approximate_num_terms=orders)
        self._lv = lv
        self.expansion = expansion
        self._n = expansion.num_terms
        self._nt = self._n
        twv = basis_mod.transverse_wavevectors(
            in_plane_wavevector=jnp.asarray(self.k_in),
            primitive_lattice_vectors=lv,
            expansion=expansion,
        )
        self._twv = twv.astype(self._cdt)
        self._omega = layer_matrices_omega(self.wavelength).astype(self._cdt)
        self._squarings = squarings_override
        # Cache the jitted core ONCE: `jax.jit(bound_method)` misses JAX's function cache when the
        # Jit is rebuilt per call, so a warm `solve()` used to re-trace/recompile (~4x wall time, the
        # stage-0 bench's 4505ms-vs-1142ms at 2n=2402). Same caching as the jNO seam's `_core_jit`.
        self._solve_core_jit = jax.jit(self._solve_core)

        # Per-layer STATIC decisions from the concrete prototype grids.
        self._thick = []
        self._shapes = []
        self._chunks = []
        self._uniform = []
        self._mixed_guard = []
        self._prototype = []
        nl = len(self.layers)
        if chunk_override is not None and len(chunk_override) < nl:
            raise ValueError(f"`chunk_override` must have one entry per layer ({nl}), got {len(chunk_override)}.")
        for li, (thick, eps_grid) in enumerate(self.layers):
            if not (math.isfinite(float(thick)) and thick > 0):
                raise ValueError(f"layer {li} thickness must be a positive finite value, got {thick!r}.")
            eps_grid = jnp.asarray(eps_grid)
            if eps_grid.ndim == 0:
                # A bare scalar permittivity means a uniform layer.
                eps_grid = jnp.full((1, 1), eps_grid)
            eps_grid = eps_grid.astype(self._cdt)
            if eps_grid.ndim != 2:
                raise ValueError(
                    f"layer {li} eps must be a scalar-compatible grid of shape "
                    f"(1, 1) or (nx, ny); got shape {eps_grid.shape}."
                )
            self._thick.append(float(thick))
            self._shapes.append(tuple(eps_grid.shape))
            uniform = tuple(eps_grid.shape) == (1, 1)
            self._uniform.append(uniform)
            self._prototype.append(eps_grid)
            rho = spectral_bound(eps_grid, self._twv, self._omega)
            nc = chunk_count(self._thick[-1], rho, theta)
            if chunk_override is not None and chunk_override[li]:
                nc = int(chunk_override[li])
            self._chunks.append(nc)
            if not uniform:
                # The guard norm is a build-time static bound; the traced core
                # recomputes E_t (and the tangent field) from the solve grids.
                _, z_eps, e_t, _ = matrices.transverse_matrices_isotropic(
                    eps_grid,
                    expansion,
                    formulation,
                    lv,
                )
                m_l = layer.assemble_M_L(self._omega, twv, z_eps, e_t)
                self._mixed_guard.append(layer.matrix_inf_norm(m_l) * (self._thick[-1] / self._chunks[-1]))
            else:
                # Keep `_mixed_guard` indexed by layer position so mixed
                # stacks (uniform layers above/below patterned ones) work.
                self._mixed_guard.append(None)

    # -- solve -------------------------------------------------------------

    def solve(
        self,
        wavelength=None,
        eps_grids=None,
        polarization=(1.0, 0.0),
        validate: bool = True,
        k_in=None,
        amb_sup=None,
        amb_sub=None,
    ):
        """Solve for plane-wave incidence at the stack's ``k_in`` (or an override).

        Args:
            wavelength: Overrides the build wavelength (may be traced).
            eps_grids: Replaces the prototype layer grids (list of ``(1, 1)``
                or ``(nx, ny)`` arrays, one per layer; may be traced). These
                are the design items and flow through the compiled core.
            polarization: ``"x"``/``"y"`` or a Jones pair.
            validate: Runs the core eagerly on concrete results first so the
                never-silent guards (expm bound, energy) fire (house style);
                structural checks run regardless.
            k_in: Overrides the build incidence wavevector ``(kx, ky)`` (may be
                traced). The chunking/squaring guards were fixed at build from
                the build ``k_in``; a shifted incidence is handled by the core
                exactly, with the build's (conservative) stability decisions.
            amb_sup, amb_sub: Overrides of the uniform ambient permittivities
                (may be traced scalars or single-grid values).

        Returns:
            A ``_Result`` with ``T``/``R`` efficiencies, per-order fluxes and
            the substrate/reflected state vectors.
        """
        pol = _polarization_vector(polarization)
        if eps_grids is None:
            grids = self._prototype
        else:
            if len(eps_grids) != len(self.layers):
                raise ValueError(
                    f"eps_grids has {len(eps_grids)} entries for {len(self.layers)} "
                    "layers -- the stack structure is fixed at build; one grid per layer."
                )
            grids = []
            for li, g in enumerate(eps_grids):
                g = jnp.asarray(g).astype(self._cdt)
                if tuple(g.shape) != self._shapes[li]:
                    raise ValueError(
                        f"eps grid {li} shape {tuple(g.shape)} does not match the "
                        f"build shape {self._shapes[li]} -- structural decisions "
                        f"are fixed at build."
                    )
                grids.append(g)
        wl = self.wavelength if wavelength is None else jnp.asarray(wavelength)
        if _concrete(wl):
            wl = jnp.asarray(wl).astype(self._omega.real.dtype)
        kin = self.k_in if k_in is None else jnp.asarray(k_in)
        amb_sup = self.amb_sup if amb_sup is None else amb_sup
        amb_sub = self.amb_sub if amb_sub is None else amb_sub

        if validate and all(_concrete(g) for g in grids) and _concrete(wl):
            # The eager pre-run exists so the expm-bound/energy guards fire before
            # the compiled core returns; it is NOT a traced execution, so the
            # counting hook (traces only) must not see it.
            before = _PATTERNED_SOLVES[0]
            res = _Result.from_tuple(self._solve_core(tuple(grids), wl, kin, pol, amb_sup, amb_sub))
            _PATTERNED_SOLVES[0] = before
            if _concrete(res.T):
                _energy_guard(res, self._chunks)
        core = self._solve_core_jit
        res = _Result.from_tuple(core(tuple(grids), wl, kin, pol, amb_sup, amb_sub))
        # The returned value comes from the COMPILED core, which for a chaotic
        # (un-chunked, c64) solve can diverge from the eager pre-run: guard the
        # final result too, so a direct backend user never receives a silently
        # unconserved answer (the seam has its own equivalent checks).
        if validate and _concrete(res.T):
            _energy_guard(res, self._chunks)
        return res

    def _solve_core(self, grids, wl, kin, pol, amb_sup, amb_sub):
        """jitted core: static chunking/squaring means no data-dependent flow.

        ``twv`` derives from the *per-solve* ``kin`` inside the trace (the seam
        sweeps the incidence angle through the compiled core), and the ambient
        split uses the per-solve ``amb_sup``/``amb_sub``. The chunk counts,
        Pade squaring counts and uniform/patterned flags stay static from the
        build, which only set stability margins -- never correctness.
        """
        n = self._n
        two_n = 2 * n
        omega = layer_matrices_omega(wl).astype(self._cdt)
        twv = basis_mod.transverse_wavevectors(
            in_plane_wavevector=jnp.asarray(kin),
            primitive_lattice_vectors=self._lv,
            expansion=self.expansion,
        ).astype(self._cdt)
        formulation = self.formulation
        lv = self._lv
        expansion = self.expansion

        transfers = []
        for li, eps_grid in enumerate(grids):
            d_c = self._thick[li] / self._chunks[li]
            # A (1, 1) grid is uniform NOW (the seam's eager collapse fires on concrete constant
            # grids at solve time), regardless of the build-time shape: the analytic shortcut is a
            # property of the per-solve permittivity, never a structure decision.
            if tuple(jnp.shape(eps_grid)) == (1, 1):
                t = layer.uniform_layer_transfer(
                    omega,
                    twv,
                    jnp.reshape(eps_grid, ()),
                    d_c,
                    dtype=self._cdt,
                )
            else:
                _PATTERNED_SOLVES[0] += 1  # counting oracle (traces only)
                _, z_eps, e_t, _ = matrices.transverse_matrices_isotropic(eps_grid, expansion, formulation, lv)
                m_l = layer.assemble_M_L(omega, twv, z_eps, e_t)
                t = layer.layer_transfer_exp(
                    m_l,
                    d_c,
                    squarings=self._squarings,
                    guard_norm=self._mixed_guard[li],
                )
            transfers.append(t)

        kz_sub = layer.ambient_kz(twv, amb_sub, omega, dtype=self._cdt)
        kz_sup = layer.ambient_kz(twv, amb_sup, omega, dtype=self._cdt)
        y_sub = layer.ambient_admittance_matrix(twv, kz_sub, omega, dtype=self._cdt)
        y_sup = layer.ambient_admittance_matrix(twv, kz_sup, omega, dtype=self._cdt)

        u_inc = jnp.zeros(two_n, dtype=self._cdt)
        u_inc = u_inc.at[0].set(-pol[1])
        u_inc = u_inc.at[n].set(pol[0])

        def blocks(t):
            return (
                t[:two_n, :two_n],
                t[:two_n, two_n:],
                t[two_n:, :two_n],
                t[two_n:, two_n:],
            )

        # 1) Impedance recursion bottom -> top, collecting the top-side
        #    admittance of every chunk in bottom->top order.
        y = y_sub
        y_stack_bottom_up = []
        for li in range(len(transfers) - 1, -1, -1):
            a, b, c, d = blocks(transfers[li])
            for _ in range(self._chunks[li]):
                y = jnp.linalg.solve(d - y @ b, y @ a - c)
                y_stack_bottom_up.append(y)
        y_stack_bottom_up.reverse()  # top-side admittance per chunk, top-first
        # A stack with no finite films (a single homogenized medium: superstrate == substrate) is
        # trivial -- the "stack" admittance is the substrate's own.
        y_stack = y_stack_bottom_up[0] if y_stack_bottom_up else y_sub

        # 2) Reflection solve at the superstrate interface.
        u_ref = jnp.linalg.solve(y_stack + y_sup, (y_sup - y_stack) @ u_inc)
        v_ref = -y_sup @ u_ref

        # 3) Stable propagation of the tangential-E state only: at the top of
        #    every chunk v = Y u, so u_bottom = (A + B Y_top) u_top. Growing
        #    evanescent modes live inside v = Y u and never appear as
        #    standalone growing quantities (the classic S-matrix stabilization,
        #    here in state space).
        u = u_inc + u_ref
        k = 0
        for li in range(len(transfers)):
            a, b, c, d = blocks(transfers[li])
            for _ in range(self._chunks[li]):
                u = (a + b @ y_stack_bottom_up[k]) @ u
                k += 1
        v = y_sub @ u

        # 4) Readouts (signed per-order Poynting, jNO convention).
        flux_inc = layer.sz_flux(u_inc, y_sup @ u_inc, n)
        flux_trn = layer.sz_flux(u, v, n)
        flux_rfl = layer.sz_flux(u_ref, v_ref, n)
        pin = jnp.sum(flux_inc)
        return (
            jnp.sum(flux_trn) / pin,
            -jnp.sum(flux_rfl) / pin,
            pin,
            flux_trn,
            flux_rfl,
            flux_inc,
            u,
            v,
            u_ref,
            v_ref,
        )

    def order_index(self, m: int, n: int) -> int:
        """The expansion index of diffraction order ``(m, n)``."""
        import numpy as onp

        bc = onp.asarray(self.expansion.basis_coefficients)
        hit = onp.where((bc[:, 0] == m) & (bc[:, 1] == n))[0]
        if not len(hit):
            raise ValueError(f"order ({m},{n}) outside the truncation")
        return int(hit[0])

    @property
    def num_terms(self) -> int:
        return self._n


class _Result:
    """The plane-wave solve result: efficiencies, fluxes and end states.

    Attributes:
        T, R: Power transmission/reflection efficiencies (lossless stacks give
            ``T + R <= 1 + 5e-3``).
        pin: Incident (zeroth-order) power, in the incident-state units.
        flux_trn, flux_rfl, flux_inc: Per-order signed Poynting fluxes, ``(n,)``.
        u_sub, v_sub: Tangential state ``(u, v) = ((-E_y, E_x), (H_x, H_y))`` at
            the substrate interface (through-put state).
        u_ref, v_ref: Reflected state at the superstrate interface.
    """

    __slots__ = (
        "T",
        "R",
        "pin",
        "flux_trn",
        "flux_rfl",
        "flux_inc",
        "u_sub",
        "v_sub",
        "u_ref",
        "v_ref",
    )

    def __init__(
        self,
        t,
        r,
        pin,
        flux_trn,
        flux_rfl,
        flux_inc,
        u_sub,
        v_sub,
        u_ref,
        v_ref,
    ):
        self.T = t
        self.R = r
        self.pin = pin
        self.flux_trn = flux_trn
        self.flux_rfl = flux_rfl
        self.flux_inc = flux_inc
        self.u_sub = u_sub
        self.v_sub = v_sub
        self.u_ref = u_ref
        self.v_ref = v_ref

    @classmethod
    def from_tuple(cls, vals):
        return cls(*vals)

    def __repr__(self):
        def _f(x):
            return f"{float(x):.6f}" if _concrete(x) else f"<{type(x).__name__}>"

        return f"_Result(T={_f(self.T)}, R={_f(self.R)})"


def layer_matrices_omega(wavelength) -> jnp.ndarray:
    """``omega = 2 pi / wavelength`` with ``c == 1`` (jNO convention)."""
    return 2 * jnp.pi / jnp.asarray(wavelength)
