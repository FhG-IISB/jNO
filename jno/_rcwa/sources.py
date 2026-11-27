"""Internal-source (dipole / Gaussian emitter) machinery for the state-space backend.

An emitter in ``jno.rcwa`` is a forcing current in the weak form, inferred by the
front door into a source spec and routed here. The genuinely-new piece (vs the
fmmax-era backend, which used Whittaker's modal emission matrix) is that the
solve is closed directly on the **Maxwell jump conditions** at the source plane,
in state space, with no eigendecomposition anywhere:

* The stack is split at the source plane into an upper (toward the superstrate)
  and lower (toward the substrate) half-stack, each chunked and transferred by
  the same ``_rcwa.stack`` machinery as a plane-wave solve.
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
* Each half-stack is linear and outgoing at its ambient: ``v = Y u`` at every
  internal plane, with the substrate-side admittance from the plane-wave
  bottom-up recursion and the superstrate-side from the same recursion pushed
  top-down from the ``v = -Y_sup u`` radiation condition. The two half-stack
  admittances ``Y_L``, ``Y_U`` at the source plane close the source-plane solve
  (a single ``2n x 2n`` linear system), and the emitted power is the flux of the
  propagated states at the two ambients.

The model is anchored against fmmax's ``amplitudes_for_source`` in the engine
(test) geometry to ~1e-14 (complex128), including off-center sources, z-dipoles,
patterned source layers, gaussian sources and the ``1/omega`` prefactor. The
Fourier-space current generators (``dirac_delta_source``, ``gaussian_source``)
are the same math as ``fmmax.sources``, ported without importing fmmax.

Everything here is jit- and grad-clean: only JAX ops, no data-dependent control
flow (chunk counts and half-stack layouts are static at build; the static inputs
``si`` and ``kind`` are Python values fixed by the front door at construction
and the jitted core is cached per ``(si, kind)``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import basis as basis_mod


def dirac_delta_source(
    location: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
    expansion: basis_mod.Expansion,
) -> jnp.ndarray:
    """Fourier coefficients of a point (delta) source at ``location``.

    ``exp(-i (kx x0 + ky y0))`` per order -- identical math to fmmax's
    ``dirac_delta_source``. ``location`` has shape ``(num_sources, 2)`` (the
    trailing axis is x, y); returns shape ``(num_terms, num_sources)``.

    Args:
        location: ``(num_sources, 2)`` x/y coordinates in the unit cell.
        in_plane_wavevector: ``(..., 2)`` k_in (batch-broadcastable).
        primitive_lattice_vectors, expansion: the lattice and truncation.

    Returns:
        Coefficients, ``(..., num_terms, num_sources)``.
    """
    if location.ndim != 2 or location.shape[-1] != 2:
        raise ValueError(f"`location` must be rank-2 with a trailing axis of 2, got shape {location.shape}")
    if in_plane_wavevector.shape[-1] != 2:
        raise ValueError(f"`in_plane_wavevector` must have a trailing axis of 2, got {in_plane_wavevector.shape}")
    twv = basis_mod.transverse_wavevectors(
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )
    kx = twv[..., 0][..., jnp.newaxis]
    ky = twv[..., 1][..., jnp.newaxis]
    x = location[:, 0]
    y = location[:, 1]
    return jnp.exp(-1j * (kx * x + ky * y))


def gaussian_source(
    fwhm: jnp.ndarray,
    location: jnp.ndarray,
    in_plane_wavevector: jnp.ndarray,
    primitive_lattice_vectors: basis_mod.LatticeVectors,
    expansion: basis_mod.Expansion,
) -> jnp.ndarray:
    """Fourier coefficients of a Gaussian source of ``fwhm`` at ``location``.

    ``exp(-i k·x0) * exp(-sigma^2 |k|^2 / 2)`` with
    ``sigma = fwhm / (2 sqrt(2 ln 2))`` -- identical math to fmmax's
    ``gaussian_source`` (the Gibbs-free soft source used by the front door for a
    Gaussian emitter).

    Args:
        fwhm: Full-width at half-maximum (scalar array).
        location: ``(num_sources, 2)`` x/y coordinates.
        in_plane_wavevector, primitive_lattice_vectors, expansion: as
            :func:`dirac_delta_source`.

    Returns:
        Coefficients, ``(..., num_terms, num_sources)``.
    """
    if location.ndim != 2 or location.shape[-1] != 2:
        raise ValueError(f"`location` must be rank-2 with a trailing axis of 2, got shape {location.shape}")
    if in_plane_wavevector.shape[-1] != 2:
        raise ValueError(f"`in_plane_wavevector` must have a trailing axis of 2, got {in_plane_wavevector.shape}")
    twv = basis_mod.transverse_wavevectors(
        in_plane_wavevector=in_plane_wavevector,
        primitive_lattice_vectors=primitive_lattice_vectors,
        expansion=expansion,
    )
    kx = twv[..., 0][..., jnp.newaxis]
    ky = twv[..., 1][..., jnp.newaxis]
    x = location[:, 0]
    y = location[:, 1]
    coeffs = jnp.exp(-1j * (kx * x + ky * y))
    sigma = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    return coeffs * jnp.exp(-(sigma**2) * (kx**2 + ky**2) / 2)


def polarization_terms(
    jx: jnp.ndarray,
    jy: jnp.ndarray,
    jz: jnp.ndarray,
    inverse_z_permittivity: jnp.ndarray,
    transverse_wavevectors: jnp.ndarray,
    omega,
) -> jnp.ndarray:
    """The source-side jump vector ``(s_u, s_v)`` across the source plane.

    Derivation (Maxwell jump conditions, state-space closure): the tangential
    current ``(jx, jy)`` jumps the tangential magnetic field, and the z-current
    ``jz`` enters through the ``E_z`` elimination (Ampere's law in the source
    layer), producing a source in the ``u`` channel proportional to the
    transverse wavevector times the inverse-rule z-permittivity; the ``1/omega``
    prefactor follows from time-harmonic Maxwell in the ``e^{-iwt}`` phasor
    convention used by the backend (anchored to fmmax at wl 1.0, 0.7, 1.55 to
    ~1e-14). This is the eig-free substitute for the Whittaker-§7 modal emission
    vector.

    Args:
        jx, jy, jz: Fourier coefficients of the x/y/z current, ``(num_terms,)``
            each (``amp * orient * source_coeffs``).
        inverse_z_permittivity: The ``(num_terms, num_terms)`` inverse-rule
            z-permittivity matrix of the source layer (as returned by the
            backend's Toeplitz builder), or a scalar ``1/eps`` for a uniform
            layer.
        transverse_wavevectors: Per-order ``(num_terms, 2)`` wavevectors.
        omega: Angular frequency ``2 pi / wavelength``.

    Returns:
        The ``(4 * num_terms,)`` jump vector ``(s_u, s_v)``, top half the u
        (E_t) channel jump, bottom half the v (H_t) channel jump.
    """
    kx = transverse_wavevectors[:, 0]
    ky = transverse_wavevectors[:, 1]
    if jnp.ndim(inverse_z_permittivity) == 0:
        eta_jz = inverse_z_permittivity * jz
    else:
        eta_jz = inverse_z_permittivity @ jz
    s_u = (1.0 / omega) * jnp.concatenate([-ky * eta_jz, kx * eta_jz])

    # s_v: the tangential-current H jump (v = (Hx, Hy)): Hx jumps by jy, Hy by -jx.
    s_v = jnp.concatenate([jy, -jx])
    return jnp.concatenate([s_u, s_v])


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

    n = stack._n
    two_n = 2 * n
    cdt = stack._cdt
    lv = stack._lv
    expansion = stack.expansion
    chunks = list(stack._chunks)
    nfilm = len(chunks)

    # ---- static half-stack chunk layouts: top-first (li, part) per chunk, where part is "full" for a
    # whole film's chunk and "up"/"lo" for the source film's slabs above/below the source plane ----
    n_up = max(1, chunks[si])
    upper_specs = []
    for li in range(si):
        upper_specs.extend([(li, "full")] * chunks[li])
    upper_specs.extend([(si, "up")] * n_up)
    n_lo = max(1, chunks[si])
    lower_specs = [(si, "lo")] * n_lo
    for li in range(si + 1, nfilm):
        lower_specs.extend([(li, "full")] * chunks[li])

    def core(grids, wl, kin, amb_sup, amb_sub, tu, tl, loc, orient, amp, fwhm, thick):
        omega = (2 * jnp.pi / jnp.asarray(wl)).astype(cdt)
        twv = basis_mod.transverse_wavevectors(
            in_plane_wavevector=jnp.asarray(kin),
            primitive_lattice_vectors=lv,
            expansion=expansion,
        ).astype(cdt)

        # source coefficients (same math as fmmax sources.py)
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

        # ---- one material record per film, one block tuple per distinct chunk thickness ----
        records = {}
        blocks = {}

        def chunk_blocks(li, part):
            """(A, B, C, D) of one chunk of film `li`: a whole-film chunk, or a slab of the source film."""
            if (li, part) not in blocks:
                if li not in records:
                    entry = grids[li] if isinstance(grids[li], tuple) else (grids[li],)
                    if len(entry) > 1:  # an anisotropic layer is outside the source solve's scope
                        raise ValueError(
                            "internal-source solves support scalar-isotropic layers "
                            f"only; got {len(entry)}-component material at layer {li}."
                        )
                    records[li] = stack._layer_material(li, grids[li], omega, twv, count=False)
                d = {"full": thick[li] / chunks[li], "up": tu / n_up, "lo": tl / n_lo}[part]
                blocks[(li, part)] = records[li].transfer(d)
            return blocks[(li, part)]

        upper, lower = upper_specs, lower_specs

        # ---- ambient admittances -------------------------------------------
        kz_sub = layer_mod.ambient_kz(twv, amb_sub, omega, dtype=cdt)
        kz_sup = layer_mod.ambient_kz(twv, amb_sup, omega, dtype=cdt)
        y_sub = layer_mod.ambient_admittance_matrix(twv, kz_sub, omega, dtype=cdt)
        y_sup = layer_mod.ambient_admittance_matrix(twv, kz_sup, omega, dtype=cdt)

        # ---- upper half-stack admittance at the source plane ----------------
        # Push top-down from the radiation condition v = -y_sup u at the top;
        # chunk transfer maps top->bottom, and at each chunk top v = y u:
        #   y_bot = (C + D y_top) (A + B y_top)^{-1}
        # The application is a RIGHT inverse, so use the conjugate-transpose solve
        # form (solve(M^H, X^H)^H == X M^{-1}) instead of an explicit `inv` -- same
        # accuracy/consistency discipline as the sibling solve sites below.
        y = -y_sup
        up_ytops = []
        for li, part in upper:
            A, B, C, D = chunk_blocks(li, part)
            up_ytops.append(y)
            y = jnp.linalg.solve((A + B @ y).conj().T, (C + D @ y).conj().T).conj().T
        YU = y if upper else -y_sup

        # ---- lower half-stack admittance at the source plane ----------------
        # Push bottom-up from the substrate: each step maps a chunk's bottom
        # admittance to its top: y_top = (D - y_bot B)^{-1}(y_bot A - C).
        y = y_sub
        lo_ytops = []
        for li, part in reversed(lower):
            A, B, C, D = chunk_blocks(li, part)
            y = jnp.linalg.solve(D - y @ B, y @ A - C)
            lo_ytops.append(y)
        lo_ytops.reverse()  # top-first
        YL = lo_ytops[0] if lo_ytops else y_sub

        # ---- source jump vector (the source film's own inverse-z eta) --------
        scr_entry = grids[si]
        scr_entry = scr_entry if isinstance(scr_entry, tuple) else (scr_entry,)
        scr = jnp.asarray(scr_entry[0]).astype(cdt)
        if tuple(jnp.shape(scr)) == (1, 1):
            eta = 1.0 / jnp.reshape(scr, ())
        else:
            eta = fft_mod.fourier_convolution_matrix(1.0 / scr, expansion)
        s = polarization_terms(jx, jy, jz, eta, twv, omega)
        s_u = s[:two_n]
        s_v = s[two_n:]

        # ---- source-plane solve --------------------------------------------
        u_L = jnp.linalg.solve(YU - YL, s_v - YU @ s_u)
        u_U = u_L + s_u

        # ---- down: propagate to the substrate (y-assisted, stable) ----------
        u = u_L
        for (li, part), yt in zip(lower, lo_ytops):
            A, B, C, D = chunk_blocks(li, part)
            u = (A + B @ yt) @ u
        v = y_sub @ u
        down = jnp.abs(jnp.sum(layer_mod.sz_flux(u, v, n)))

        # ---- up: propagate from the source plane to the superstrate ---------
        u = u_U
        for idx in range(len(upper) - 1, -1, -1):
            li, part = upper[idx]
            A, B, C, D = chunk_blocks(li, part)
            yt = up_ytops[idx]
            u = jnp.linalg.solve(A + B @ yt, u)
        v = -y_sup @ u
        up = jnp.abs(jnp.sum(layer_mod.sz_flux(u, v, n)))

        return up, down

    return jax.jit(core)
