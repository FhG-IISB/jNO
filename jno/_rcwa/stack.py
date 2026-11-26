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


def spectral_bound(eps_layer, twv, omega) -> float:
    """Static bound on ``max|q| = sqrt(|omega^2 eps| + |k+G|^2)`` (host-side).

    ``eps_layer`` may be a single grid or a tuple of anisotropic component
    grids; the max-|eps| runs over every component (the zz/transverse entries
    all contribute to the per-order spectral norm via the same bound).
    """
    import numpy as onp

    comps = eps_layer if isinstance(eps_layer, (tuple, list)) else (eps_layer,)
    max_eps = 0.0
    for c in comps:
        a = onp.asarray(c)
        max_eps = max(max_eps, float(onp.max(onp.abs(a))))
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


def chunk_blocks(t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """The four ``n x n`` blocks ``(A, B, C, D)`` of a ``4n x 4n`` chunk transfer."""
    two_n = t.shape[-1] // 2
    return (
        t[:two_n, :two_n],
        t[:two_n, two_n:],
        t[two_n:, :two_n],
        t[two_n:, two_n:],
    )


class _LayerRecord:
    """Traced material record of one layer for the interior-field core.

    ``transfer(d)`` is the 4n x 4n chunk transfer at thickness ``d`` (the
    same-material expm/analytic builder the plane-wave solve uses at the
    chunk thickness, evaluated at any sub-chunk depth); ``blocks`` is the
    chunk-transfer block tuple at the chunk thickness (the impedance
    recursion). ``z_eps`` / ``z_mu`` are the inverse-rule zz Fourier matrices
    of a patterned layer (``None`` for uniform / analytic layers, where
    per-order scalar division applies); ``eps_zz`` / ``mu_zz`` are the scalar
    zz components used by the uniform paths and the analytic ambients.
    """

    __slots__ = ("mode", "eps_zz", "mu_zz", "z_eps", "z_mu", "transfer", "blocks")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class StateStack:
    """Eig-free state-space RCWA stack with the jNO seam shape.

    Args:
        wavelength: Free-space wavelength (the build-time value; may be
            overridden per :meth:`solve`).
        period: Lattice periods ``(Px, Py)``.
        amb_sup, amb_sub: Superstrate and substrate permittivities (uniform).
        layers: The stack ``[(thickness, eps), ...]`` from top to bottom.
            `eps` is a scalar or permittivity grid for an isotropic layer, a
            5-tuple ``(eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)`` for an
            anisotropic epsilon layer, or a 10-tuple adding
            ``(mu_xx, mu_xy, mu_yx, mu_yy, mu_zz)`` for a general (magnetic)
            anisotropic layer. Tuple components may be scalars or grids.
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
        self._ncomp = []  # 1 (isotropic), 5 or 10 (anisotropic)
        self._aniso = []  # bool: a tensor layer (5/10 components)
        self._mu = []  # bool: a 10-component (magnetic) layer
        self._comp_shapes = []  # per-layer tuple of per-component shapes
        self._uniform = []
        self._chunks = []
        self._mixed_guard = []
        self._prototype = []
        nl = len(self.layers)
        # One entry per layer POSITION (0 for a non-aniso layer), not one per
        # aniso layer: the traced consumer at `_layer_material` indexes by the
        # ABSOLUTE layer index. Appending only per-aniso-layer breaks mixed
        # iso/aniso stacks (IndexError on iso-above-aniso; silently-shifted
        # squaring counts when an iso layer sits above two aniso layers) --
        # the same position-indexing family as `_mixed_guard` (iss_mwurllzvaj6pf4).
        self._aniso_squarings = [0] * nl
        if chunk_override is not None and len(chunk_override) < nl:
            raise ValueError(f"`chunk_override` must have one entry per layer ({nl}), got {len(chunk_override)}.")

        def _normalize_components(eps):
            """A layer's material into ``(ncomp, [component grids], is_mu)``.

            Accepts a scalar / ``(1, 1)`` / ``(nx, ny)`` grid (isotropic) or a
            5-tuple ``(eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)`` /
            10-tuple adding ``(mu_xx, ...)`` (anisotropic), with scalars
            promoted to ``(1, 1)`` grids. Tuples of other lengths raise loudly.
            """
            if isinstance(eps, tuple):
                n = len(eps)
                if n not in (5, 10):
                    raise ValueError(f"a tuple layer material must have 5 (eps) or 10 (eps+mu) components, got {n}.")
                comps = []
                for c in eps:
                    c = jnp.asarray(c)
                    if c.ndim == 0:
                        c = jnp.full((1, 1), c)
                    comps.append(c)
                return n, comps, n == 10
            g = jnp.asarray(eps)
            if g.ndim == 0:
                g = jnp.full((1, 1), g)
            return 1, [g], False

        for li, (thick, eps_layer) in enumerate(self.layers):
            if not (math.isfinite(float(thick)) and thick > 0):
                raise ValueError(f"layer {li} thickness must be a positive finite value, got {thick!r}.")
            ncomp, comps, is_mu = _normalize_components(eps_layer)
            comps = [c.astype(self._cdt) for c in comps]
            shapes = [tuple(c.shape) for c in comps]
            if any(len(s) != 2 for s in shapes) or any(min(s) < 1 for s in shapes):
                raise ValueError(f"layer {li} component grids must be 2-D (1, 1) or (nx, ny); got shapes {shapes}.")
            uniform = all(s == (1, 1) for s in shapes)
            # Never-silent tensor guard on the CONCRETE prototype values: a
            # singular / negative-definite tensor is caught at build (and, for a
            # per-solve collapse, again under solve's eager validation).
            if ncomp > 1 and all(_concrete(c) for c in comps):
                layer.validate_anisotropic_tensor(
                    tuple(comps[:5]), tuple(comps[5:]) if is_mu else None, li, uniform=uniform
                )
            self._thick.append(float(thick))
            self._ncomp.append(ncomp)
            self._aniso.append(ncomp in (5, 10))
            self._mu.append(is_mu)
            self._comp_shapes.append(tuple(shapes))
            self._uniform.append(uniform)
            self._prototype.append(tuple(comps))
            rho = spectral_bound(comps, self._twv, self._omega)
            nc = chunk_count(self._thick[-1], rho, theta)
            if chunk_override is not None and chunk_override[li]:
                nc = int(chunk_override[li])
            self._chunks.append(nc)
            if not uniform and ncomp == 1:
                # The guard norm is a build-time static bound; the traced core
                # recomputes E_t (and the tangent field) from the solve grids.
                _, z_eps, e_t, _ = matrices.transverse_matrices_isotropic(
                    comps[0],
                    expansion,
                    formulation,
                    lv,
                )
                m_l = layer.assemble_M_L(self._omega, twv, z_eps, e_t)
                self._mixed_guard.append(layer.matrix_inf_norm(m_l) * (self._thick[-1] / self._chunks[-1]))
            else:
                # The uniform-aniso fast path needs a STATIC Pade squaring count
                # for its per-order transfers, fixed at build from a spectral
                # bound of the per-order 4x4 blocks of the prototype (the same
                # stability-margin discipline as the patterned expm guard). This
                # is needed for ANY aniso layer: a build-PATTERNED tensor layer
                # that a per-solve concrete collapse makes uniform must still
                # hand the traced core a static count.
                if ncomp > 1:
                    d_c = self._thick[-1] / self._chunks[-1]
                    norm = layer.uniform_aniso_spectral_norm(
                        tuple(comps[:5]),
                        tuple(comps[5:]) if is_mu else None,
                        twv,
                        self._omega,
                        d_c,
                        self._cdt,
                    )
                    self._aniso_squarings[li] = layer.squarings_for_spectral_norm(norm)
                if not uniform:  # a patterned anisotropic layer
                    mu_comps = comps[5:] if is_mu else None
                    (_, z_eps, e_t, _, z_mu, m_t, _) = matrices.transverse_matrices_anisotropic(
                        tuple(comps[:5]),
                        tuple(mu_comps) if mu_comps is not None else None,
                        expansion,
                        formulation,
                        lv,
                    )
                    m_l = layer.assemble_M_L(self._omega, twv, z_eps, e_t, transverse_permeability=m_t, z_permeability=z_mu)
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
                    "layers -- the stack structure is fixed at build; one material per layer."
                )
            grids = []
            for li, g in enumerate(eps_grids):
                entry = g if isinstance(g, tuple) else (g,)
                comps = []
                for c in entry:
                    c = jnp.asarray(c).astype(self._cdt)
                    if c.ndim == 0:
                        c = jnp.full((1, 1), c)
                    comps.append(c)
                shapes = [tuple(c.shape) for c in comps]
                if tuple(shapes) != self._comp_shapes[li] and tuple(shapes) != ((1, 1),):
                    raise ValueError(
                        f"material {li} shapes {tuple(shapes)} does not match the "
                        f"build shapes {self._comp_shapes[li]} -- structural decisions "
                        f"are fixed at build."
                    )
                grids.append(tuple(comps) if len(comps) > 1 else comps[0])
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

    def _layer_material(self, li, layer_grid, omega, twv, count: bool):
        """The traced material record of layer `li` for the per-solve grids.

        Returns a :class:`_LayerRecord` whose ``transfer(d)`` computes the
        4n x 4n chunk transfer at any thickness ``d <= d_c`` (the per-plane
        interior probe reuses the chunk's own builder at sub-chunk depths),
        whose ``blocks()`` are the chunk-transfer blocks at ``d_c`` (the
        impedance recursion), and which carries ``z_eps``/``z_mu`` (``None``
        when uniform) for the algebraic ez/hz readouts.

        ``count`` gates the patterned-eigensolve counting oracle: the plane-
        wave solve counts patterned expms, the interior-field core does not
        (it would double-count the same transfers it re-derives).

        ``uniform_now`` is a per-solve SHAPE decision (static under trace);
        the uniform shortcut is a property of the per-solve permittivity,
        never a structure decision.
        """
        d_c = self._thick[li] / self._chunks[li]
        entry = layer_grid if isinstance(layer_grid, tuple) else (layer_grid,)
        comps = [jnp.asarray(c).astype(self._cdt) for c in entry]
        shapes = tuple(tuple(jnp.shape(c)) for c in comps)
        uniform_now = shapes == ((1, 1),) or all(s == (1, 1) for s in shapes)
        if len(comps) > 1:  # an anisotropic (tensor) layer
            if uniform_now:
                if count:
                    _PATTERNED_SOLVES[0] += 0  # analytic per-order path, not a patterned expm
                mu_comps = comps[5:] if len(comps) == 10 else None
                eps_scalars = tuple(jnp.reshape(c, ()) for c in comps[:5])
                mu_scalars = tuple(jnp.reshape(c, ()) for c in mu_comps) if mu_comps is not None else None
                squarings = self._aniso_squarings[li] if self._aniso_squarings else 0

                def transfer(d, omega=omega, twv=twv, eps=eps_scalars, mu=mu_scalars, s=squarings):
                    return layer.uniform_anisotropic_layer_transfer(omega, twv, eps, mu, d, squarings=s, dtype=self._cdt)

                return _LayerRecord(
                    mode="uniform-aniso",
                    eps_zz=eps_scalars[4],
                    mu_zz=mu_scalars[4] if mu_scalars is not None else jnp.asarray(1.0, dtype=self._cdt),
                    z_eps=None,
                    z_mu=None,
                    transfer=transfer,
                    blocks=chunk_blocks(transfer(d_c)),
                )
            if count:
                _PATTERNED_SOLVES[0] += 1  # counting oracle (traces only)
            mu_comps = comps[5:] if len(comps) == 10 else None
            (_, z_eps, e_t, _, z_mu, m_t, _) = matrices.transverse_matrices_anisotropic(
                tuple(comps[:5]),
                tuple(mu_comps) if mu_comps is not None else None,
                self.expansion,
                self.formulation,
                self._lv,
            )
            m_l = layer.assemble_M_L(omega, twv, z_eps, e_t, transverse_permeability=m_t, z_permeability=z_mu)
            guard_norm = self._mixed_guard[li]
            squarings = self._squarings

            def transfer(d, m_l=m_l, guard_norm=guard_norm, squarings=squarings):
                return layer.layer_transfer_exp(m_l, d, squarings=squarings, guard_norm=guard_norm)

            return _LayerRecord(
                mode="patterned",
                eps_zz=None,
                mu_zz=None,
                z_eps=z_eps,
                z_mu=z_mu,
                transfer=transfer,
                blocks=chunk_blocks(transfer(d_c)),
            )
        if uniform_now:
            eps = jnp.reshape(comps[0], ())

            def transfer(d, omega=omega, twv=twv, eps=eps):
                return layer.uniform_layer_transfer(omega, twv, eps, d, dtype=self._cdt)

            return _LayerRecord(
                mode="uniform",
                eps_zz=eps,
                mu_zz=jnp.asarray(1.0, dtype=self._cdt),
                z_eps=None,
                z_mu=None,
                transfer=transfer,
                blocks=chunk_blocks(transfer(d_c)),
            )
        if count:
            _PATTERNED_SOLVES[0] += 1  # counting oracle (traces only)
        _, z_eps, e_t, _ = matrices.transverse_matrices_isotropic(comps[0], self.expansion, self.formulation, self._lv)
        m_l = layer.assemble_M_L(omega, twv, z_eps, e_t)
        guard_norm = self._mixed_guard[li]
        squarings = self._squarings

        def transfer(d, m_l=m_l, guard_norm=guard_norm, squarings=squarings):
            return layer.layer_transfer_exp(m_l, d, squarings=squarings, guard_norm=guard_norm)

        return _LayerRecord(
            mode="patterned",
            eps_zz=None,
            mu_zz=None,
            z_eps=z_eps,
            z_mu=None,
            transfer=transfer,
            blocks=chunk_blocks(transfer(d_c)),
        )

    def _solve_states(self, omega, twv, records, amb_sup, amb_sub, pol):
        """The impedance recursion, reflection solve and stable chunk sweep.

        Shared by the plane-wave core (which reads T/R off the end states) and
        the interior-field core (which also keeps every chunk-top state). This
        is exactly the state-space solve of the original ``_solve_core``,
        unchanged numerically.

        Returns:
            A dict with ``kz_sup/kz_sub``, ``y_sup/y_sub``, ``u_inc``,
            ``u_ref/v_ref``, the substrate end state ``u_sub/v_sub``, the
            per-chunk top-side admittances ``y_stack_bottom_up`` (top-first)
            and the per-chunk top states ``chunk_top``.
        """
        n = self._n
        two_n = 2 * n
        kz_sub = layer.ambient_kz(twv, amb_sub, omega, dtype=self._cdt)
        kz_sup = layer.ambient_kz(twv, amb_sup, omega, dtype=self._cdt)
        y_sub = layer.ambient_admittance_matrix(twv, kz_sub, omega, dtype=self._cdt)
        y_sup = layer.ambient_admittance_matrix(twv, kz_sup, omega, dtype=self._cdt)

        u_inc = jnp.zeros(two_n, dtype=self._cdt)
        u_inc = u_inc.at[0].set(-pol[1])
        u_inc = u_inc.at[n].set(pol[0])

        # 1) Impedance recursion bottom -> top, collecting the top-side
        #    admittance of every chunk in bottom->top order.
        y = y_sub
        y_stack_bottom_up = []
        for li in range(len(records) - 1, -1, -1):
            a, b, c, d = records[li].blocks
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
        chunk_top = []
        k = 0
        for li in range(len(records)):
            a, b, c, d = records[li].blocks
            for _ in range(self._chunks[li]):
                chunk_top.append(u)
                u = (a + b @ y_stack_bottom_up[k]) @ u
                k += 1
        v_sub = y_sub @ u

        return {
            "kz_sup": kz_sup,
            "kz_sub": kz_sub,
            "y_sup": y_sup,
            "y_sub": y_sub,
            "u_inc": u_inc,
            "u_ref": u_ref,
            "v_ref": v_ref,
            "u_sub": u,
            "v_sub": v_sub,
            "y_stack_bottom_up": y_stack_bottom_up,
            "chunk_top": chunk_top,
        }

    def _solve_core(self, grids, wl, kin, pol, amb_sup, amb_sub):
        """jitted core: static chunking/squaring means no data-dependent flow.

        ``twv`` derives from the *per-solve* ``kin`` inside the trace (the seam
        sweeps the incidence angle through the compiled core), and the ambient
        split uses the per-solve ``amb_sup``/``amb_sub``. The chunk counts,
        Pade squaring counts and uniform/patterned flags stay static from the
        build, which only set stability margins -- never correctness.
        """
        n = self._n
        omega = layer_matrices_omega(wl).astype(self._cdt)
        twv = basis_mod.transverse_wavevectors(
            in_plane_wavevector=jnp.asarray(kin),
            primitive_lattice_vectors=self._lv,
            expansion=self.expansion,
        ).astype(self._cdt)
        records = [self._layer_material(li, layer_grid, omega, twv, count=True) for li, layer_grid in enumerate(grids)]
        st = self._solve_states(omega, twv, records, amb_sup, amb_sub, pol)

        # 4) Readouts (signed per-order Poynting, jNO convention).
        flux_inc = layer.sz_flux(st["u_inc"], st["y_sup"] @ st["u_inc"], n)
        flux_trn = layer.sz_flux(st["u_sub"], st["v_sub"], n)
        flux_rfl = layer.sz_flux(st["u_ref"], st["v_ref"], n)
        pin = jnp.sum(flux_inc)
        return (
            jnp.sum(flux_trn) / pin,
            -jnp.sum(flux_rfl) / pin,
            pin,
            flux_trn,
            flux_rfl,
            flux_inc,
            st["u_sub"],
            st["v_sub"],
            st["u_ref"],
            st["v_ref"],
        )

    # ------------------------------------------------------------------
    # Interior field reconstruction (stage 2d): per-plane Fourier fields.
    # ------------------------------------------------------------------

    def interior_field_amplitudes(self, grids, wl, kin, pol, amb_sup, amb_sub, planes):
        """Per-plane Fourier field amplitudes inside the stack.

        `planes` is a static tuple of per-plane specs (Python values, resolved
        eagerly by ``jno/_rcwa/fields.py`` from the concrete z layout):

        * ``("sup", offset)`` -- analytic superstrate plane, `offset` the depth
          below the superstrate interface (<= 0, so the incident phase
          ``exp(i kz_sup * offset)`` is referenced to the interface);
        * ``("sub", z)`` -- analytic substrate plane at depth `z` below the
          substrate interface;
        * ``("film", film_idx, chunk, zeta)`` -- inside film `film_idx`, `zeta`
          below the top of its `chunk`.

        The field core re-derives the layer transfers and the impedance
        recursion exactly as the plane-wave solve does, then propagates the
        boundary state to each plane via ``u = (A + B Y_top) u_top`` -- the
        stable per-chunk form, so evanescent-heavy interior probes never form
        standalone growing quantities.

        Returns:
            ``(ex, ey, ez, hx, hy, hz)``, each ``(P, n)`` complex Fourier
            coefficients per order, in plane order.
        """
        fn = self.__dict__.get("_field_core_jit")
        if fn is None:
            fn = self._field_core_jit = jax.jit(self._field_core, static_argnums=(6,))
        return fn(
            tuple(grids),
            jnp.asarray(wl),
            jnp.asarray(kin),
            tuple(jnp.asarray(p) for p in pol),
            jnp.asarray(amb_sup),
            jnp.asarray(amb_sub),
            tuple(planes),
        )

    def _field_core(self, grids, wl, kin, pol, amb_sup, amb_sub, planes):
        """jitted interior-field core: ``(ex, ey, ez, hx, hy, hz)`` per plane.

        `planes` is static (a Python tuple baked into the trace), so the
        compiled program has no data-dependent indexing: every plane's layer,
        chunk and depth are constants, and the only traced inputs are the
        materials (design), wavelength, incidence and ambients.
        """
        n = self._n
        cdt = self._cdt
        omega = layer_matrices_omega(wl).astype(cdt)
        twv = basis_mod.transverse_wavevectors(
            in_plane_wavevector=jnp.asarray(kin),
            primitive_lattice_vectors=self._lv,
            expansion=self.expansion,
        ).astype(cdt)
        records = [self._layer_material(li, layer_grid, omega, twv, count=False) for li, layer_grid in enumerate(grids)]
        st = self._solve_states(omega, twv, records, amb_sup, amb_sub, pol)
        # global chunk offset of each layer's first chunk (film planes give
        # per-layer chunk indices; the recursion lists every chunk top-first).
        chunk_starts = []
        _acc = 0
        for c in self._chunks:
            chunk_starts.append(_acc)
            _acc += c
        kx = twv[:, 0]
        ky = twv[:, 1]
        e_om = 1.0 / omega
        y_stack_bottom_up = st["y_stack_bottom_up"]
        chunk_top = st["chunk_top"]

        def _rep(ph):  # per-order phase -> the (u1 | u2) component-major state order
            return jnp.concatenate([ph, ph])

        exs, eys, ezs, hxs, hys, hzs = [], [], [], [], [], []
        for spec in planes:
            if spec[0] == "sup":
                ph_f = _rep(jnp.exp(1j * st["kz_sup"] * spec[1]))
                ph_b = _rep(jnp.exp(-1j * st["kz_sup"] * spec[1]))
                u = st["u_inc"] * ph_f + st["u_ref"] * ph_b
                v = st["y_sup"] @ (st["u_inc"] * ph_f - st["u_ref"] * ph_b)
                eps_zz = amb_sup
                mu_zz = jnp.asarray(1.0, dtype=cdt)
            elif spec[0] == "sub":
                ph = _rep(jnp.exp(1j * st["kz_sub"] * spec[1]))
                u = st["u_sub"] * ph
                v = st["y_sub"] @ u
                eps_zz = amb_sub
                mu_zz = jnp.asarray(1.0, dtype=cdt)
            else:
                rec = records[spec[1]]
                a, b, c_block, d = chunk_blocks(rec.transfer(spec[3]))
                gk = chunk_starts[spec[1]] + spec[2]
                y_top = y_stack_bottom_up[gk]
                u_top = chunk_top[gk]
                u = (a + b @ y_top) @ u_top
                v = (c_block + d @ y_top) @ u_top
                eps_zz = rec.eps_zz
                mu_zz = rec.mu_zz
            ex = u[n:]
            ey = -u[:n]
            hx = v[:n]
            hy = v[n:]
            if eps_zz is None:  # patterned layer: the inverse-rule Fourier solve
                ez = jnp.linalg.solve(rec.z_eps, -(kx * hy - ky * hx) * e_om)
                if rec.z_mu is not None:
                    hz = jnp.linalg.solve(rec.z_mu, (kx * ey - ky * ex) * e_om)
                else:
                    hz = (kx * ey - ky * ex) * e_om
            else:  # uniform / uniform-anisotropic / analytic ambient
                ez = -(kx * hy - ky * hx) * e_om / eps_zz
                hz = (kx * ey - ky * ex) * e_om / mu_zz
            exs.append(ex)
            eys.append(ey)
            ezs.append(ez)
            hxs.append(hx)
            hys.append(hy)
            hzs.append(hz)
        return (
            jnp.stack(exs, axis=0),
            jnp.stack(eys, axis=0),
            jnp.stack(ezs, axis=0),
            jnp.stack(hxs, axis=0),
            jnp.stack(hys, axis=0),
            jnp.stack(hzs, axis=0),
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
