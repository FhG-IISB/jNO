"""The stack solve: every film one S-matrix, one sweep through them.

* Each film is split into same-material sub-slabs (chunks) at BUILD time; the
  chunk count is a static integer from the concrete spectral bound
  ``|q|_max * d`` (``q`` = modal wavenumber, bounded by
  ``sqrt(|omega^2 eps|_max + max|k+G|^2)``), rounded up to a power of two. A
  chunk's growth is then at most ``e``, so its transfer (``layer.py``) is
  accurate and well conditioned.
* The chunk transfer becomes a symmetric S-matrix in one fixed lossy reference
  medium, which is doubled up to the film (``jno/_rcwa/smatrix.py``). The film
  S-matrices stay bounded however thick or evanescent the film is.
* One sweep from the substrate up gives the reflection in front of every film,
  ``R_j = r + t R_{j+1} (I - r R_{j+1})^-1 t`` (Li's R-matrix recursion on
  S-matrices); the superstrate closes the problem and the downward amplitude is
  carried back down. The only analytic plane-wave splits are at the two
  ambients: per-order ``kz = sqrt(eps*omega^2 - |k+G|^2)`` with the
  ``Im(kz) >= 0`` branch and the per-order admittances.
* Interior fields run the same sweep over the films cut at the plane depths;
  internal sources (``sources.py``) run it from both ambients to the source
  plane.

Readouts follow jNO's conventions exactly: signed per-order Poynting
``S_z = 0.5 Re(u1 conj(v1) + u2 conj(v2))`` summed over orders, ``T`` from the
transmitted state, ``R`` from the reflected state with the sign flip,
normalized by the incident power.

Jit hygiene: chunk counts, series doubling counts and uniform/patterned flags
are static Python values decided at build from the concrete prototype grids and
thicknesses; the grids and the thicknesses themselves flow through pure JAX
inside the traced core (native JVPs). There are no data-dependent loops in the
compiled program. How far a solve's grids and thicknesses are from what the
static counts were sized for is returned as ``load`` (<= 1 is within budget).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from . import basis as basis_mod
from . import layer, matrices, smatrix
from .matrices import Formulation

#: Per-chunk growth measure kept <= this by the build-time chunking.
_CHUNK_THETA = 1.0
#: House energy guard, exactly jNO's rcwa convention.
_ENERGY_TOL = 5e-3

_DEFAULT_FORMULATION = Formulation.JONES_DIRECT_FOURIER


def _concrete(x) -> bool:
    return not isinstance(x, jax.core.Tracer)


def _load_guard(res):
    """Never-silent check that the solve stayed inside the static budget fixed at build (eager only).

    ``res.load`` is the largest over the layers of the thin-slice series bound ``||P h0^2|| / theta`` and of
    ``(d / (THICKNESS_MARGIN d_build))^2``: above 1, a thickness or a permittivity has moved further from the
    build prototype than the static doubling and chunk counts allow for.
    """
    load = float(res.load)
    if not load <= 1.0 + 1e-9:
        raise ValueError(
            f"_rcwa: a layer left the range this stack was built for (load {load:.3g} > 1): a thickness beyond "
            f"{layer.THICKNESS_MARGIN:g}x the built one, or a permittivity much larger than the build grid's. "
            "Build a fresh stack (a new Rcwa) at the new thickness or design."
        )


def _largest_components(comps, is_mu):
    """The per-component largest-magnitude values of a tensor layer, as ``(eps5, mu5 or None)`` scalars.

    They size the static doubling count of the per-order uniform path for any constant the grids may
    collapse to within the prototype's range (host-side)."""
    import numpy as onp

    def big(c):
        a = onp.asarray(c).reshape(-1)
        return complex(a[onp.argmax(onp.abs(a))])

    vals = [big(c) for c in comps]
    return tuple(vals[:5]), (tuple(vals[5:]) if is_mu else None)


def _energy_guard(res, auto_chunks):
    """The never-silent energy guard (house style): rejects NaN/inf efficiencies,
    ``T + R > 1 + tol`` (lossless stacks) and negative efficiencies with an
    actionable message. Eager host-side only -- callers gate on ``_concrete``.

    ``auto_chunks`` is the build-time per-layer chunk count, reported so the
    T+R>1 cliff (an under-chunked layer) points at its remedy.
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
    """The number of patterned-layer transfers traced since the last reset."""
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
    """Static same-material sub-slab count: ``ceil(rho * d / theta)`` rounded up to a power of two, so that the
    plane-wave solve builds the layer's S-matrix from the chunk's by doublings alone."""
    c = max(1, int(math.ceil(rho * thickness / theta)))
    return 1 << (c - 1).bit_length()


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

    ``transfer(d)`` is the block tuple ``(A, B, C, D)`` of the transfer over
    a depth ``d`` (the builder the plane-wave solve uses at the chunk
    thickness, evaluated at any sub-chunk depth); ``blocks`` is that tuple at
    the chunk thickness (the chunk S-matrix) and ``load`` the layer's
    build-budget diagnostic there. ``z_eps`` / ``z_mu`` are the inverse-rule zz Fourier matrices
    of a patterned layer (``None`` for uniform / analytic layers, where
    per-order scalar division applies); ``eps_zz`` / ``mu_zz`` are the scalar
    zz components used by the uniform paths and the analytic ambients.
    """

    __slots__ = ("mode", "eps_zz", "mu_zz", "z_eps", "z_mu", "transfer", "blocks", "load")

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
        doublings_override: Optional fixed series doubling count for all
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
        doublings_override: Optional[int] = None,
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
        self._doublings = []  # per layer position: the static series doubling count of the patterned path
        self._prototype = []
        nl = len(self.layers)
        # One entry per layer POSITION (0 for a non-aniso layer): the traced consumer at
        # `_layer_material` indexes by the ABSOLUTE layer index (iss_mwurllzvaj6pf4).
        self._uniform_doublings = [0] * nl
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
            d_c = self._thick[-1] / self._chunks[-1]
            k_patterned = 0
            if not uniform:
                # A build-time static bound from the prototype; the traced core rebuilds B and C from the
                # solve grids and reports in `load` how close they come to it.
                b_mat, c_mat, _, _ = self._patterned_bc(comps, self._omega, self._twv)
                k_patterned = layer.series_doublings(layer.matrix_inf_norm(b_mat @ c_mat), d_c)
            if ncomp > 1:
                # A tensor layer whose solve grids collapse to constants takes the per-order uniform path,
                # which needs its own static count (sized on the prototype's largest component values).
                eps5, mu5 = _largest_components(comps, is_mu)
                b_u, c_u = layer.uniform_aniso_bc(self._omega, self._twv, eps5, mu5, dtype=self._cdt)
                self._uniform_doublings[li] = layer.series_doublings(layer.matrix_inf_norm(b_u @ c_u), d_c)
            if doublings_override is not None:
                k_patterned = int(doublings_override)
            self._doublings.append(k_patterned)

    def _patterned_bc(self, comps, omega, twv):
        """``(B, C, z_eps, z_mu)`` of a patterned layer from its component grids (one grid when isotropic)."""
        if len(comps) > 1:
            mu_comps = comps[5:] if len(comps) == 10 else None
            (_, z_eps, e_t, _, z_mu, m_t, _) = matrices.transverse_matrices_anisotropic(
                tuple(comps[:5]),
                tuple(mu_comps) if mu_comps is not None else None,
                self.expansion,
                self.formulation,
                self._lv,
            )
            b, c = layer.layer_bc(omega, twv, z_eps, e_t, transverse_permeability=m_t, z_permeability=z_mu)
            return b, c, z_eps, z_mu
        _, z_eps, e_t, _ = matrices.transverse_matrices_isotropic(comps[0], self.expansion, self.formulation, self._lv)
        b, c = layer.layer_bc(omega, twv, z_eps, e_t)
        return b, c, z_eps, None

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
        thick=None,
    ):
        """Solve for plane-wave incidence at the stack's ``k_in`` (or an override).

        Args:
            wavelength: Overrides the build wavelength (may be traced).
            eps_grids: Replaces the prototype layer grids (list of ``(1, 1)``
                or ``(nx, ny)`` arrays, one per layer; may be traced). These
                are the design items and flow through the compiled core.
            polarization: ``"x"``/``"y"`` or a Jones pair.
            validate: On a concrete result, raises if energy is not conserved or
                a layer left the build budget (``load > 1``); structural checks
                run regardless.
            k_in: Overrides the build incidence wavevector ``(kx, ky)`` (may be
                traced). The chunk and doubling counts were fixed at build from
                the build ``k_in``; a shifted incidence is handled by the core
                exactly, with the build's (conservative) stability decisions.
            amb_sup, amb_sub: Overrides of the uniform ambient permittivities
                (may be traced scalars or single-grid values).
            thick: Per-layer thicknesses overriding the build ones (may be
                traced; the static counts allow up to ``THICKNESS_MARGIN`` x
                the build thickness, see ``load``).

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

        thick = tuple(self._thick) if thick is None else tuple(thick)
        res = _Result.from_tuple(self._solve_core_jit(tuple(grids), wl, kin, pol, amb_sup, amb_sub, thick))
        if validate and _concrete(res.T):
            _energy_guard(res, self._chunks)
            _load_guard(res)
        return res

    def _layer_material(self, li, layer_grid, omega, twv, count: bool, d_c=None):
        """The traced material record of layer `li` for the per-solve grids.

        Returns a :class:`_LayerRecord` whose ``transfer(d)`` gives the four
        ``2n x 2n`` blocks of the transfer over any depth ``d`` (the chunk,
        or a sub-chunk depth for an interior plane), whose ``blocks`` are
        those at the chunk thickness ``d_c`` (the chunk S-matrix; ``None``
        when ``d_c`` is not given), whose ``load`` is the layer's share of the
        solve's build-budget diagnostic at ``d_c``, and which carries
        ``z_eps``/``z_mu`` (``None`` when uniform) for the ez/hz readouts.

        ``count`` gates the patterned-solve counting oracle: the plane-wave
        solve counts patterned layers, the interior-field core does not.

        ``uniform_now`` is a per-solve SHAPE decision (static under trace);
        the uniform shortcut is a property of the per-solve permittivity,
        never a structure decision.
        """
        entry = layer_grid if isinstance(layer_grid, tuple) else (layer_grid,)
        comps = [jnp.asarray(c).astype(self._cdt) for c in entry]
        shapes = tuple(tuple(jnp.shape(c)) for c in comps)
        uniform_now = shapes == ((1, 1),) or all(s == (1, 1) for s in shapes)
        degree = layer.series_degree(self._cdt)
        d_c_build = self._thick[li] / self._chunks[li]

        def thickness_load(d):
            return (jnp.abs(jnp.real(d)) / (layer.THICKNESS_MARGIN * d_c_build)) ** 2

        if len(comps) > 1 and uniform_now:  # a uniform anisotropic (tensor) layer: per-order 4x4 transfers
            mu_comps = comps[5:] if len(comps) == 10 else None
            eps_scalars = tuple(jnp.reshape(c, ()) for c in comps[:5])
            mu_scalars = tuple(jnp.reshape(c, ()) for c in mu_comps) if mu_comps is not None else None
            k = self._uniform_doublings[li]
            b_u, c_u = layer.uniform_aniso_bc(omega, twv, eps_scalars, mu_scalars, dtype=self._cdt)
            p_u = b_u @ c_u

            def transfer(d, omega=omega, twv=twv, eps=eps_scalars, mu=mu_scalars, k=k):
                return chunk_blocks(
                    layer.uniform_anisotropic_layer_transfer(omega, twv, eps, mu, d, doublings=k, dtype=self._cdt)
                )

            def load(d, p_u=p_u, k=k):
                return jnp.maximum(layer.series_load(p_u, d, k), thickness_load(d))

            rec = dict(
                mode="uniform-aniso",
                eps_zz=eps_scalars[4],
                mu_zz=mu_scalars[4] if mu_scalars is not None else jnp.asarray(1.0, dtype=self._cdt),
                z_eps=None,
                z_mu=None,
            )
        elif uniform_now:  # a uniform isotropic layer: the closed form
            eps = jnp.reshape(comps[0], ())

            def transfer(d, omega=omega, twv=twv, eps=eps):
                return chunk_blocks(layer.uniform_layer_transfer(omega, twv, eps, d, dtype=self._cdt))

            load = thickness_load
            rec = dict(mode="uniform", eps_zz=eps, mu_zz=jnp.asarray(1.0, dtype=self._cdt), z_eps=None, z_mu=None)
        else:  # a patterned layer, isotropic or tensor: the thin-slice series in P = B C
            if count:
                _PATTERNED_SOLVES[0] += 1  # counting oracle (traces only)
            b, c, z_eps, z_mu = self._patterned_bc(comps, omega, twv)
            p = b @ c
            k = self._doublings[li]

            def transfer(d, b=b, c=c, p=p, k=k):
                return layer.slice_transfer(b, c, p, d, k, degree)

            def load(d, p=p, k=k):
                return jnp.maximum(layer.series_load(p, d, k), thickness_load(d))

            rec = dict(mode="patterned", eps_zz=None, mu_zz=None, z_eps=z_eps, z_mu=z_mu)
        return _LayerRecord(
            transfer=transfer,
            blocks=None if d_c is None else transfer(d_c),
            load=None if d_c is None else load(d_c),
            **rec,
        )

    def _slab(self, rec, li, thickness, chunks: int, y_r, z_r):
        """The symmetric S-matrix ``(r, t)`` of film ``li`` over ``thickness`` (may be traced): the transfer over
        ``thickness / chunks`` turned into a chunk S-matrix, then doubled up (``chunks`` static)."""
        blocks = rec.blocks if thickness is None else rec.transfer(thickness / chunks)
        return smatrix.layer_smatrix(*smatrix.chunk_smatrix(blocks, y_r, z_r), chunks)

    def _gap_chunks(self, li, gap: float) -> int:
        """Static chunk count for a sub-thickness ``gap`` of film ``li``: no chunk thicker than the film's own
        build chunk (so its series and growth bounds hold), rounded up to a power of two."""
        c = max(1, int(math.ceil(gap / (self._thick[li] / self._chunks[li]) - 1e-9)))
        return 1 << (c - 1).bit_length()

    def _frame(self, wl, kin, amb_sup, amb_sub):
        """Per-solve ``omega``, wavevectors, the reference medium and the two ambients' per-order admittances."""
        cdt = self._cdt
        omega = layer_matrices_omega(wl).astype(cdt)
        twv = basis_mod.transverse_wavevectors(
            in_plane_wavevector=jnp.asarray(kin),
            primitive_lattice_vectors=self._lv,
            expansion=self.expansion,
        ).astype(cdt)
        y_r, z_r = smatrix.reference(twv, omega, cdt)
        kz_sup = layer.ambient_kz(twv, amb_sup, omega, dtype=cdt)
        kz_sub = layer.ambient_kz(twv, amb_sub, omega, dtype=cdt)
        y_sup = layer.ambient_admittance(twv, kz_sup, omega, dtype=cdt)
        y_sub = layer.ambient_admittance(twv, kz_sub, omega, dtype=cdt)
        return dict(omega=omega, twv=twv, y_r=y_r, z_r=z_r, kz_sup=kz_sup, kz_sub=kz_sub, y_sup=y_sup, y_sub=y_sub)

    def _incident(self, pol):
        n = self._n
        return jnp.zeros(2 * n, dtype=self._cdt).at[0].set(-pol[1]).at[n].set(pol[0])

    def _solve_core(self, grids, wl, kin, pol, amb_sup, amb_sub, thick=None):
        """jitted core: static chunk and doubling counts mean no data-dependent flow.

        Every film is one slab: its S-matrix in the reference modes, built from a chunk's by doublings. The sweep
        from the substrate up gives the stack's reflection, the superstrate closes the problem, and the downward
        amplitude is carried to the substrate (``jno/_rcwa/smatrix.py``).

        ``twv`` derives from the *per-solve* ``kin`` inside the trace (the seam
        sweeps the incidence angle through the compiled core), and the ambient
        split uses the per-solve ``amb_sup``/``amb_sub``. The chunk counts,
        series doubling counts and uniform/patterned flags stay static from the
        build, which only set stability margins -- never correctness; ``thick``
        (default: the build thicknesses) may be traced, and the returned
        ``load`` says how far the solve is from those margins.
        """
        n = self._n
        fr = self._frame(wl, kin, amb_sup, amb_sub)
        y_r, z_r, y_sup, y_sub = fr["y_r"], fr["z_r"], fr["y_sup"], fr["y_sub"]
        records = self._records(grids, fr["omega"], fr["twv"], thick, count=True)
        load = jnp.max(jnp.stack([jnp.real(r.load) for r in records])) if records else jnp.asarray(0.0)

        slabs = [self._slab(rec, li, None, self._chunks[li], y_r, z_r) for li, rec in enumerate(records)]
        rho_sub = smatrix.ambient_reflection(y_r, y_sub)
        rs, xs = smatrix.sweep(slabs, smatrix.dense(rho_sub))
        u_inc = self._incident(pol)
        u_ref, a_top = smatrix.reflect_plane_wave(rs[0], y_sup, z_r, u_inc)
        a_bot = smatrix.carry(xs, a_top)[-1]
        u_sub = a_bot + smatrix.apply(rho_sub, a_bot)
        v_ref = -smatrix.apply(y_sup, u_ref)
        v_sub = smatrix.apply(y_sub, u_sub)

        # Readouts (signed per-order Poynting, jNO convention).
        flux_inc = layer.sz_flux(u_inc, smatrix.apply(y_sup, u_inc), n)
        flux_trn = layer.sz_flux(u_sub, v_sub, n)
        flux_rfl = layer.sz_flux(u_ref, v_ref, n)
        pin = jnp.sum(flux_inc)
        return (
            jnp.sum(flux_trn) / pin,
            -jnp.sum(flux_rfl) / pin,
            pin,
            flux_trn,
            flux_rfl,
            flux_inc,
            u_sub,
            v_sub,
            u_ref,
            v_ref,
            load,
        )

    def _records(self, grids, omega, twv, thick, count):
        """Per-layer material records at the per-solve chunk thickness ``thick[li] / chunks[li]``."""
        thick = self._thick if thick is None else thick
        return [
            self._layer_material(li, g, omega, twv, count=count, d_c=thick[li] / self._chunks[li])
            for li, g in enumerate(grids)
        ]

    # ------------------------------------------------------------------
    # Interior field reconstruction: per-plane Fourier fields.
    # ------------------------------------------------------------------

    def interior_field_amplitudes(self, grids, wl, kin, pol, amb_sup, amb_sub, planes, thick=None):
        """Per-plane Fourier field amplitudes inside the stack.

        `planes` is a static tuple of per-plane specs (Python values, resolved
        eagerly by ``jno/_rcwa/fields.py`` from the concrete z layout):

        * ``("sup", offset)`` -- analytic superstrate plane, `offset` the depth
          below the superstrate interface (<= 0, so the incident phase
          ``exp(i kz_sup * offset)`` is referenced to the interface);
        * ``("sub", z)`` -- analytic substrate plane at depth `z` below the
          substrate interface;
        * ``("film", film_idx, zeta)`` -- inside film `film_idx`, `zeta` below
          its top.

        Every film is cut at its plane depths into slabs, and the same sweep as
        the plane-wave solve runs over that finer sequence: the state at a cut
        is ``a+`` carried down to it and ``a- = R a+`` with the reflection of
        everything below, so evanescent-heavy interior probes never form
        growing quantities. The thicknesses are concrete here (``thick``,
        default the build ones) and part of the compiled program.

        Returns:
            ``(ex, ey, ez, hx, hy, hz)``, each ``(P, n)`` complex Fourier
            coefficients per order, in plane order.
        """
        fn = self.__dict__.get("_field_core_jit")
        if fn is None:
            fn = self._field_core_jit = jax.jit(self._field_core, static_argnums=(6, 7))
        thick = self._thick if thick is None else thick
        return fn(
            tuple(grids),
            jnp.asarray(wl),
            jnp.asarray(kin),
            tuple(jnp.asarray(p) for p in pol),
            jnp.asarray(amb_sup),
            jnp.asarray(amb_sub),
            tuple(planes),
            tuple(float(jnp.real(jnp.asarray(t))) for t in thick),
        )

    def _field_core(self, grids, wl, kin, pol, amb_sup, amb_sub, planes, thick):
        """jitted interior-field core: ``(ex, ey, ez, hx, hy, hz)`` per plane.

        `planes` and `thick` are static (Python values baked into the trace),
        so every cut, slab and chunk count is a constant, and the only traced
        inputs are the materials (design), wavelength, incidence and ambients.
        """
        n = self._n
        cdt = self._cdt
        fr = self._frame(wl, kin, amb_sup, amb_sub)
        y_r, z_r, y_sup, y_sub = fr["y_r"], fr["z_r"], fr["y_sup"], fr["y_sub"]
        omega, twv = fr["omega"], fr["twv"]
        records = self._records(grids, omega, twv, thick, count=False)

        # Cut every film at its plane depths; key the cuts so that a plane finds its interface.
        def key(z, t):
            return round(min(max(z, 0.0), t) / t, 12)

        slabs, iface, idx, cache = [], {}, 0, {}
        for f, (rec, t) in enumerate(zip(records, thick)):
            cuts = sorted({0.0, 1.0} | {key(spec[2], t) for spec in planes if spec[0] == "film" and spec[1] == f})
            iface[(f, cuts[0])] = idx
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                gap = (hi - lo) * t
                k = (f, round(gap, 12))
                if k not in cache:
                    cache[k] = self._slab(rec, f, gap, self._gap_chunks(f, gap), y_r, z_r)
                slabs.append(cache[k])
                idx += 1
                iface[(f, hi)] = idx

        rho_sub = smatrix.ambient_reflection(y_r, y_sub)
        rs, xs = smatrix.sweep(slabs, smatrix.dense(rho_sub))
        u_inc = self._incident(pol)
        u_ref, a_top = smatrix.reflect_plane_wave(rs[0], y_sup, z_r, u_inc)
        a_plus = smatrix.carry(xs, a_top)
        u_sub = a_plus[-1] + smatrix.apply(rho_sub, a_plus[-1])

        kx = twv[:, 0]
        ky = twv[:, 1]
        e_om = 1.0 / omega

        def _rep(ph):  # per-order phase -> the (u1 | u2) component-major state order
            return jnp.concatenate([ph, ph])

        exs, eys, ezs, hxs, hys, hzs = [], [], [], [], [], []
        for spec in planes:
            if spec[0] == "sup":
                ph_f = _rep(jnp.exp(1j * fr["kz_sup"] * spec[1]))
                ph_b = _rep(jnp.exp(-1j * fr["kz_sup"] * spec[1]))
                u = u_inc * ph_f + u_ref * ph_b
                v = smatrix.apply(y_sup, u_inc * ph_f - u_ref * ph_b)
                eps_zz = amb_sup
                mu_zz = jnp.asarray(1.0, dtype=cdt)
            elif spec[0] == "sub":
                u = u_sub * _rep(jnp.exp(1j * fr["kz_sub"] * spec[1]))
                v = smatrix.apply(y_sub, u)
                eps_zz = amb_sub
                mu_zz = jnp.asarray(1.0, dtype=cdt)
            else:
                rec = records[spec[1]]
                j = iface[(spec[1], key(spec[2], thick[spec[1]]))]
                u, v = smatrix.state(a_plus[j], rs[j] @ a_plus[j], y_r)
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
        load: The build-budget diagnostic (``<= 1``: the thicknesses and grids
            are within what the static doubling and chunk counts were sized for).
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
        "load",
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
        load=0.0,
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
        self.load = load

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
