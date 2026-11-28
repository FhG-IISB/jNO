"""Internal-source (dipole / Gaussian emitter) machinery for the state-space backend.

An emitter in ``jno.rcwa`` is a forcing current in the weak form, inferred by the
front door into a source spec and routed here. The genuinely-new piece (vs the
fmmax-era backend, which used Whittaker's modal emission matrix) is that the
solve is closed directly on the **Maxwell jump conditions** at the source plane,
in state space, with no eigendecomposition anywhere:

* The stack is split at the source plane into an upper (toward the superstrate)
  and lower (toward the substrate) half-stack, each a sequence of film S-matrices
  in the reference modes (``_rcwa.smatrix``), as in a plane-wave solve.
* Maxwell's tangential jump conditions for a surface current ``j`` at the split
  plane become, in the state ``psi = (u, v) = ((-E_y, E_x), (H_x, H_y))``:

      E_t continuous:      the *u* channel jumps only through the z-current,
                           u_U - u_L = (1/omega) * (-ky * eta @ jz,  kx * eta @ jz)
      H_t jumps by j_t:    v_U - v_L = (jy, -jx)

  where ``eta`` is the inverse-rule z-permittivity matrix of the source layer
  (``1/eps`` for a uniform layer), ``(kx, ky)`` the per-order transverse
  wavevectors and ``omega`` the angular frequency. ``u_U``/``v_U`` are the
  states at the source plane on the superstrate side, ``u_L``/``v_L`` on the
  substrate side.
* Each half-stack is outgoing at its ambient. The sweep from the substrate up
  gives the reflection ``R_L`` seen downward from the source plane
  (``a- = R_L a+``), the sweep from the superstrate down gives ``R_U`` seen
  upward (``a+ = R_U a-``). In the reference modes the jump is
  ``a_U - a_L = W^-1 s``, which with the two reflections is one ``2n x 2n``
  solve; the emitted powers are the fluxes of the amplitudes carried to the two
  ambients.

The emitted powers agreed with fmmax's ``amplitudes_for_source`` to ~1e-14
(complex128) on off-centre sources, z-dipoles, patterned source layers, Gaussian
sources and the ``1/omega`` prefactor, when that library was the test oracle.

Everything here is jit- and grad-clean: only JAX ops, no data-dependent control
flow (chunk counts and half-stack layouts are static at build; the static inputs
``si`` and ``kind`` are Python values fixed by the front door at construction
and the jitted core is cached per ``(si, kind)``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import basis as basis_mod


def _phases(location, in_plane_wavevector, primitive_lattice_vectors, expansion):
    """``exp(-i k_mn . r_0)`` per order and source, ``(..., num_terms, num_sources)``, and the ``|k_mn|^2``."""
    if location.ndim != 2 or location.shape[-1] != 2:
        raise ValueError(f"`location` must be rank-2 with a trailing axis of 2, got shape {location.shape}")
    if in_plane_wavevector.shape[-1] != 2:
        raise ValueError(f"`in_plane_wavevector` must have a trailing axis of 2, got {in_plane_wavevector.shape}")
    k = basis_mod.transverse_wavevectors(in_plane_wavevector, primitive_lattice_vectors, expansion)
    kx, ky = k[..., 0, None], k[..., 1, None]
    return jnp.exp(-1j * (kx * location[:, 0] + ky * location[:, 1])), kx**2 + ky**2


def dirac_delta_source(location, in_plane_wavevector, primitive_lattice_vectors, expansion) -> jnp.ndarray:
    """Floquet-Bloch coefficients of a point source ``delta(r - r_0)`` in the cell: ``exp(-i k_mn . r_0)``.

    (The periodised delta ``sum_R delta(r - r_0 - R) e^{i k_in . R}`` expands as ``(1/A) sum_mn exp(i k_mn . (r -
    r_0))``; the cell area ``A`` is a common factor of every emitted power and is left out, as in the readouts.)

    Args:
        location: ``(num_sources, 2)`` positions ``(x_0, y_0)`` in the cell.
        in_plane_wavevector: ``(..., 2)`` ``k_in``.
        primitive_lattice_vectors, expansion: The lattice and the truncation.

    Returns:
        ``(..., num_terms, num_sources)``.
    """
    return _phases(location, in_plane_wavevector, primitive_lattice_vectors, expansion)[0]


def gaussian_source(fwhm, location, in_plane_wavevector, primitive_lattice_vectors, expansion) -> jnp.ndarray:
    """Coefficients of a Gaussian spot of full width at half maximum ``fwhm`` at ``r_0``: the point-source phases
    times the Gaussian's spectrum ``exp(-sigma^2 |k|^2 / 2)``, ``sigma = fwhm / (2 sqrt(2 ln 2))`` (a soft source,
    free of the Gibbs ringing of a truncated delta). Shapes as :func:`dirac_delta_source`."""
    phases, k2 = _phases(location, in_plane_wavevector, primitive_lattice_vectors, expansion)
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    return phases * jnp.exp(-(sigma**2) * k2 / 2)


def polarization_terms(jx, jy, jz, inverse_z_permittivity, transverse_wavevectors, omega) -> jnp.ndarray:
    """The jump ``(s_u, s_v)`` of the state ``(u, v)`` across a current sheet ``j delta(z - z_0)``.

    With ``e^{-i omega t}`` and ``c = eps_0 = mu_0 = 1``, Ampere's law is ``curl H = -i omega eps E + J``:

    * its transverse part across the sheet: the tangential ``H`` jumps by the tangential current, ``H_x`` by ``j_y``
      and ``H_y`` by ``-j_x``, so ``s_v = (j_y, -j_x)``;
    * its z part gives ``E_z = (i k_x H_y - i k_y H_x - J_z) / (-i omega eps)``, whose ``delta`` term
      ``eta j_z / (i omega)`` (``eta`` the inverse-rule ``[[eps_zz]]^-1`` of the source layer, or ``1/eps`` when it
      is uniform) enters Faraday's law ``curl E = i omega mu H`` as ``d/dy`` and ``d/dx`` of a delta: ``E_y`` jumps
      by ``k_y eta j_z / omega`` and ``E_x`` by ``-k_x eta j_z / omega``. In ``u = (-E_y, E_x)``,
      ``s_u = (1/omega) (-k_y eta j_z, k_x eta j_z)``.

    Args:
        jx, jy, jz: Current coefficients per order, ``(num_terms,)``.
        inverse_z_permittivity: ``(num_terms, num_terms)`` matrix or a scalar.
        transverse_wavevectors: ``(num_terms, 2)``.
        omega: ``2 pi / wavelength``.

    Returns:
        ``(s_u, s_v)`` concatenated, ``(4 num_terms,)``.
    """
    kx, ky = transverse_wavevectors[:, 0], transverse_wavevectors[:, 1]
    eta_jz = inverse_z_permittivity * jz if jnp.ndim(inverse_z_permittivity) == 0 else inverse_z_permittivity @ jz
    return jnp.concatenate([(1.0 / omega) * jnp.concatenate([-ky * eta_jz, kx * eta_jz]), jy, -jx])


# -----------------------------------------------------------------------------
# The state-space emission solve on top of a StateStack (read-only use).
# -----------------------------------------------------------------------------


def _concrete(x) -> bool:
    return not isinstance(x, jax.core.Tracer)


def emission_solve(
    stack,
    source,
    wavelength=None,
    eps_grids=None,
    k_in=None,
    amb_sup=None,
    amb_sub=None,
    validate: bool = True,
    thick=None,
):
    """Internal-source (dipole / Gaussian) emission solve on a built ``StateStack``.

    The stack is split at the source plane (``source["layer"]`` = backend film
    index of the source film, ``t_upper``/``t_lower`` its distances to the film's
    top/bottom), the two half-stacks are solved exactly as plane-wave sub-stacks,
    and the jump conditions above close the source-plane state; readouts are the
    powers radiated up / into the substrate. Differentiable in ``amp``,
    ``orient``, ``loc``, ``t_upper``/``t_lower`` (z-position), ``fwhm`` and the
    design permittivity grids.

    Args:
        stack: A built :class:`jno._rcwa.stack.StateStack`.
        source: dict with keys ``layer`` (backend film index, a Python int),
            ``t_upper``, ``t_lower``, ``kind`` ('delta'|'gaussian', a str),
            ``loc`` ``(1, 2)``, ``orient`` ``(3,)``, ``amp``, ``fwhm``
            (gaussian only). ``layer`` and ``kind`` are static (fixed by the
            front door at construction); the rest may be traced.
        wavelength: Free-space wavelength (defaults to the stack's build value).
        eps_grids: Per-film permittivity grids (defaults to the stack's
            prototype grids); traced design grids are allowed.
        k_in: Transverse wavevector override.
        amb_sup, amb_sub: Ambient permittivity overrides.
        validate: Raise on a non-finite concrete emission.
        thick: Per-film thicknesses overriding the build ones (the source film's
            own split is ``t_upper``/``t_lower``).

    Returns:
        ``(up, down)``: the power radiated into the superstrate and substrate
        (both >= 0; in the source's un-normalised units, as the fmmax-era
        readouts were).
    """
    import math

    # ---- static source geometry guards (never silent) ---------------------
    si = source["layer"]
    if not (isinstance(si, int) and not isinstance(si, bool)) or not (0 <= si < len(stack.layers)):
        raise ValueError(f"source film index must be a backend layer index 0..{len(stack.layers) - 1}, got {si!r}.")
    kind = source["kind"]
    if kind not in ("delta", "gaussian"):
        raise ValueError(f"source kind must be 'delta' or 'gaussian', got {kind!r}.")
    tu = jnp.asarray(source["t_upper"])
    tl = jnp.asarray(source["t_lower"])
    if _concrete(tu) and (float(tu) < 0 or float(tl) < 0):
        raise ValueError("source t_upper/t_lower must be >= 0 (the source sits inside the source film).")

    grids = stack._prototype if eps_grids is None else list(eps_grids)
    if len(grids) != len(stack.layers):
        raise ValueError(f"eps_grids has {len(grids)} entries for {len(stack.layers)} backend layers.")
    wl = stack.wavelength if wavelength is None else jnp.asarray(wavelength)
    if _concrete(wl):
        wl = jnp.asarray(wl).astype(stack._omega.real.dtype)
    kin = stack.k_in if k_in is None else jnp.asarray(k_in)
    amb_sup = stack.amb_sup if amb_sup is None else amb_sup
    amb_sub = stack.amb_sub if amb_sub is None else amb_sub

    # The jitted core is cached ONCE per (si, kind) on the stack (same
    # discipline as _solve_core).
    cache = getattr(stack, "_emission_cores", None)
    if cache is None:
        cache = stack._emission_cores = {}
    key = (si, kind)
    if key not in cache:
        cache[key] = _make_emission_core(stack, si, kind)

    core = cache[key]
    res = core(
        tuple(grids),
        wl,
        kin,
        amb_sup,
        amb_sub,
        tu,
        tl,
        source["loc"],
        source["orient"],
        source["amp"],
        source.get("fwhm", 0.0),
        tuple(jnp.asarray(t) for t in (stack._thick if thick is None else thick)),
    )
    up, down = res
    if validate and _concrete(up):
        if not (math.isfinite(float(up)) and math.isfinite(float(down))):
            raise ValueError(
                f"non-finite emission (up={float(up):.4e} down={float(down):.4e}): a degenerate or "
                "singular source solve -- check the source layer's permittivity / location and that "
                "k_in is not exactly at a Wood/Rayleigh singularity."
            )
    return up, down


def _make_emission_core(stack, si, kind):
    """The jitted emission core: ``si``/``kind`` and the half-stack layout are
    static (baked into the closure at first emission solve); grids, wavelength,
    k_in, ambs and the source parameters are traced."""
    from . import fft as fft_mod
    from . import layer as layer_mod
    from . import smatrix

    n = stack._n
    two_n = 2 * n
    cdt = stack._cdt
    lv = stack._lv
    expansion = stack.expansion
    chunks = list(stack._chunks)
    nfilm = len(chunks)

    def core(grids, wl, kin, amb_sup, amb_sub, tu, tl, loc, orient, amp, fwhm, thick):
        omega = (2 * jnp.pi / jnp.asarray(wl)).astype(cdt)
        twv = basis_mod.transverse_wavevectors(
            in_plane_wavevector=jnp.asarray(kin),
            primitive_lattice_vectors=lv,
            expansion=expansion,
        ).astype(cdt)

        # Floquet-Bloch coefficients of the point or Gaussian source
        loc = jnp.asarray(loc).reshape(1, 2)
        kin_arr = jnp.asarray(kin)
        if kind == "gaussian":
            base = gaussian_source(jnp.asarray(fwhm), loc, kin_arr, lv, expansion)
        else:
            base = dirac_delta_source(loc, kin_arr, lv, expansion)
        base = base[:, 0]  # (n,) per-order coefficients
        orient = jnp.asarray(orient).reshape(3)
        amp = jnp.asarray(amp + 0.0j)
        jx, jy, jz = amp * orient[0] * base, amp * orient[1] * base, amp * orient[2] * base

        # ---- the two half-stacks as slabs in the reference modes, split at the source plane ----
        fr = stack._frame(wl, kin, amb_sup, amb_sub)
        y_r, z_r = fr["y_r"], fr["z_r"]
        records = []
        for li in range(nfilm):
            entry = grids[li] if isinstance(grids[li], tuple) else (grids[li],)
            if len(entry) > 1:  # an anisotropic layer is outside the source solve's scope
                raise ValueError(
                    "internal-source solves support scalar-isotropic layers "
                    f"only; got {len(entry)}-component material at layer {li}."
                )
            records.append(stack._layer_material(li, grids[li], omega, twv, count=False))

        def slab(li, d):
            return stack._slab(records[li], li, d, chunks[li], y_r, z_r)

        upper = [slab(li, thick[li]) for li in range(si)] + [slab(si, tu)]  # top first
        lower = [slab(si, tl)] + [slab(li, thick[li]) for li in range(si + 1, nfilm)]  # top first

        # Reflections at the source plane: looking down, everything below closed by the substrate (a- = R_L a+);
        # looking up, everything above closed by the superstrate (a+ = R_U a-).
        rho_sub = smatrix.ambient_reflection(y_r, fr["y_sub"])
        rho_sup = smatrix.ambient_reflection(y_r, fr["y_sup"])
        rs_lo, xs_lo = smatrix.sweep(lower, smatrix.dense(rho_sub))
        rs_up, xs_up = smatrix.sweep(upper[::-1], smatrix.dense(rho_sup))  # from the source plane upward
        r_l, r_u = rs_lo[0], rs_up[0]

        # ---- source jump vector (the source film's own inverse-z eta) --------
        scr_entry = grids[si]
        scr_entry = scr_entry if isinstance(scr_entry, tuple) else (scr_entry,)
        scr = jnp.asarray(scr_entry[0]).astype(cdt)
        if tuple(jnp.shape(scr)) == (1, 1):
            eta = 1.0 / jnp.reshape(scr, ())
        else:
            eta = fft_mod.fourier_convolution_matrix(1.0 / scr, expansion)
        s = polarization_terms(jx, jy, jz, eta, twv, omega)

        # ---- source-plane solve: psi_U - psi_L = s, i.e. a_U - a_L = W^-1 s = (sig+, sig-) ----------------
        # With a-_L = R_L a+_L and a+_U = R_U a-_U: (I - R_U R_L) a+_L = R_U sig- - sig+.
        zs_v = smatrix.apply(z_r, s[two_n:])
        sig_p, sig_m = 0.5 * (s[:two_n] + zs_v), 0.5 * (s[:two_n] - zs_v)
        eye = jnp.eye(two_n, dtype=cdt)
        a_plus_l = jnp.linalg.solve(eye - r_u @ r_l, r_u @ sig_m - sig_p)
        a_minus_u = r_l @ a_plus_l + sig_m

        # ---- down: carry a+ to the substrate; up: carry a- to the superstrate ----
        a_bot = smatrix.carry(xs_lo, a_plus_l)[-1]
        u = a_bot + smatrix.apply(rho_sub, a_bot)
        down = jnp.abs(jnp.sum(layer_mod.sz_flux(u, smatrix.apply(fr["y_sub"], u), n)))
        a_top = smatrix.carry(xs_up, a_minus_u)[-1]
        u = a_top + smatrix.apply(rho_sup, a_top)
        up = jnp.abs(jnp.sum(layer_mod.sz_flux(u, -smatrix.apply(fr["y_sup"], u), n)))

        return up, down

    return jax.jit(core)
