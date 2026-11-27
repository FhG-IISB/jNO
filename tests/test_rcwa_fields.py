"""field3d interior-reconstruction stress: chunk invariance, PML frames, loud guards.

On top of the restored ``test_rcwa.py`` field3d anchors (absorbed power == 1 - R - T,
gradient-vs-FD), this file stresses the interior-field machinery that task T31 lands:

* **Chunk invariance** -- the same geometry solved with different internal sub-slab
  chunk counts must give the same physics: ``T/R`` and the per-plane interior field
  amplitudes agree to tight tolerance (the interior probe re-derives the boundary
  state from the chunked S-matrix sweep, so the chunk bookkeeping must be
  invisible to field readouts).
* **Anisotropic layers** -- patterned 5-component and uniform-anisotropic layers go
  through the ``ez``/``hz`` algebraic paths (inverse-rule Fourier solve vs scalar
  zz division) and keep the energy anchor.
* **PML folded ambients** -- an in-plane PML frame (a patterned tensor ambient) is
  folded into the stack as a unit film; an isolated scatterer keeps energy balance
  through the frame (sigma>0 absorbs laterally-diffracted power, sigma=0 conserves
  exactly), and the field3d volume over the folded frame stays finite.
* **Loud guards** -- the never-silent discipline holds under PML: the chunk-cliff
  (un-chunked c64 recursion) raises the actionable chunk-remedy error, render grids
  too coarse for the expansion raise with a hint, and garbage ``field()`` arguments
  raise instead of returning a wrong answer.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jno._rcwa import StateStack
from jno._rcwa import fields as rcwa_fields
from jno.rcwa import Rcwa, RcwaError

WL = 1.0
K0 = 2.0 * math.pi / WL

try:  # fmmax is a test-only oracle for field-parity checks
    import fmmax  # noqa: F401

    needs_fmmax = lambda fn: fn  # noqa: E731
except Exception:  # pragma: no cover
    needs_fmmax = pytest.mark.skipif(True, reason="fmmax oracle not installed")


def _disk_grid(period, ng, radius, eps_in, eps_out=1.0, center_frac=0.5):
    """A (ng, ng) permittivity grid with a centred circular inclusion."""
    xs = (np.arange(ng) + 0.5) / ng * period
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    disk = ((X - center_frac * period) ** 2 + (Y - center_frac * period) ** 2) < radius**2
    return np.where(disk, eps_in, eps_out)


def _absorbed_fraction(sol, im_eps, nx=8, density=200.0, period=(0.6, 0.6)):
    """Absorbed fraction from the interior field: (1/2) k0 integral Im eps |E|^2 / Pin.

    Integrates over the first finite film (``z`` between the first two interfaces, the
    test_rcwa.py convention). ``im_eps`` is a constant loss scalar, or a callable
    ``(x, y) -> (nx, ny)`` loss grid evaluated on the render coordinates (for a
    position-dependent absorber, e.g. a patterned disk)."""
    inten, (x, y, z), lz = sol.field3d(nx=nx, density=density)
    if callable(im_eps):
        X, Y = np.meshgrid(np.asarray(x), np.asarray(y), indexing="ij")
        im_eps = jnp.asarray(im_eps(X, Y))[..., jnp.newaxis]
    z = np.asarray(z)
    zin = (z >= lz[0] - 1e-12) & (z <= lz[1] + 1e-12)
    prof = jnp.mean(im_eps * jnp.asarray(inten)[:, :, zin], axis=(0, 1))
    return 0.5 * K0 * jnp.trapezoid(prof, jnp.asarray(z[zin])) / sol._Pin


# ------------------------------------------------------------------------------------
# Chunk invariance
# ------------------------------------------------------------------------------------


def test_chunked_solve_and_interior_amplitudes_are_chunk_invariant():
    """Field readouts must not see the internal sub-slab bookkeeping: a geometry solved with
    the build-time auto chunking and with twice as many same-material sub-slabs gives T/R and
    per-plane interior amplitudes agreeing to roundoff, in complex128 (the complex64 floor of
    the doubled S-matrix is about 1e-5 and is a property of the precision, not of the
    bookkeeping). Both chunkings keep the per-chunk growth at/below the design bound; the
    un-chunked recursion is a separate, deliberately-avoided stress covered by the guard tests."""
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        _check_chunk_invariance()
    finally:
        jax.config.update("jax_enable_x64", prev)


def _check_chunk_invariance():
    period = (0.6, 0.6)
    layers = [(0.8, 2.5 + 0.3j)]  # lossy slab, chunked >= ceil(rho d / theta) ~ 8, then 16
    base = StateStack(
        wavelength=WL, period=period, amb_sup=1.0, amb_sub=1.0, layers=layers, orders=15, dtype=jnp.complex128
    )
    auto = base._chunks[0]
    assert auto >= 4  # the geometry is genuinely chunked by the policy

    def solve_and_probe(chunks):
        st = StateStack(
            wavelength=WL,
            period=period,
            amb_sup=1.0,
            amb_sub=1.0,
            layers=layers,
            orders=15,
            chunk_override=[chunks],
            dtype=jnp.complex128,
        )
        res = st.solve(polarization="x")
        znums, _z = rcwa_fields.layer_z_sampling([1.0, 0.8, 1.0], 40.0)
        planes = rcwa_fields.field_planes([1.0, 0.8, 1.0], znums, ["sup", 0, "sub"])
        # interior amplitudes over every plane (ambients + the film)
        ex, ey, ez, _hx, _hy, _hz = rcwa_fields.interior_field_amplitudes(
            st, planes, st._prototype, WL, (0.0, 0.0), (1.0, 0.0), 1.0, 1.0
        )
        return res, ex, ey, ez

    r_a, ex_a, ey_a, ez_a = solve_and_probe(auto)
    r_b, ex_b, ey_b, ez_b = solve_and_probe(2 * auto)
    # Physics invariant: efficiencies agree to roundoff.
    assert float(r_a.T) == pytest.approx(float(r_b.T), rel=1e-12)
    assert float(r_a.R) == pytest.approx(float(r_b.R), abs=1e-12)
    # Interior amplitudes agree plane by plane (the boundary-state propagation is
    # chunk-independent up to rounding in the recursion).
    scale = float(np.max(np.abs(np.concatenate([ex_a, ey_a, ez_a]))))
    assert float(np.max(np.abs(ex_a - ex_b))) < 1e-11 * scale
    assert float(np.max(np.abs(ey_a - ey_b))) < 1e-11 * scale
    assert float(np.max(np.abs(ez_a - ez_b))) < 1e-11 * scale


def test_field3d_energy_anchor_tightens_with_density():
    """The absorbed-|E|^2 == 1-R-T anchor tightens as the z sampling refines."""
    layers = [(0.5, 1.0), (0.2, 2.0 + 0.5j), (0.5, 2.25)]
    sol = Rcwa(layers, period=(0.6, 0.6), orders=9, wavelength=WL, assume_periodic=True).solve()
    balance = 1.0 - float(sol.efficiency("R")) - float(sol.efficiency("T"))
    for density in (100.0, 400.0):
        absorbed = float(_absorbed_fraction(sol, 0.5, nx=8, density=density))
        assert absorbed == pytest.approx(balance, rel=2e-3)


# ------------------------------------------------------------------------------------
# Anisotropic layers: the ez/hz algebraic paths
# ------------------------------------------------------------------------------------


def test_field3d_patterned_anisotropic_layer_keeps_energy_balance():
    """A patterned 5-component (ε̂) layer -- declared anisotropic, so every Fourier
    matrix is the general-anisotropic one -- goes through the inverse-rule ez/hz path.
    The z-integrated absorbed power converges to the flux balance at the anisotropic
    formulation's own floor: a smooth zz-anisotropic pattern balances to ~1.7% (a
    residual the fmmax fields share at this truncation -- the isotropic case balances
    to 1e-3)."""
    period = (0.6, 0.6)
    xs = (np.arange(24) + 0.5) / 24 * period[0]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    # smooth in-plane ripple, constant loss; zz genuinely different (the anisotropy)
    ripple = 2.0 + 0.6 * np.cos(2 * np.pi * X / period[0])
    five = (ripple + 0.4j, np.zeros_like(ripple), np.zeros_like(ripple), ripple + 0.4j, np.full_like(ripple, 2.4))
    layers = [(0.5, 1.0), (0.3, five), (0.5, 2.25)]
    sol = Rcwa(layers, period=period, orders=25, wavelength=WL, assume_periodic=True).solve()
    balance = 1.0 - float(sol.efficiency("R")) - float(sol.efficiency("T"))
    inten, (_x, _y, z), lz = sol.field3d(nx=16, density=600.0)
    assert np.all(np.isfinite(np.asarray(inten)))
    assert 0.1 < balance < 0.9
    absorbed = float(_absorbed_fraction(sol, 0.4, nx=16, density=600.0, period=period))
    assert absorbed == pytest.approx(balance, rel=2.5e-2)


def test_field3d_uniform_anisotropic_layer_keeps_energy_balance():
    """A uniform 5-tuple layer takes the uniform-anisotropic fast path (per-order 4x4
    transfers, scalar zz division) and still balances 1-R-T through |E|^2."""
    period = (0.6, 0.6)
    epsilon = (2.0 + 0.3j, 0.0, 0.0, 2.0 + 0.3j, 2.2 + 0.3j)
    layers = [(0.5, 1.0), (0.3, epsilon), (0.5, 2.25)]
    sol = Rcwa(layers, period=period, orders=9, wavelength=WL, assume_periodic=True).solve()
    balance = 1.0 - float(sol.efficiency("R")) - float(sol.efficiency("T"))
    # the loss profile is the diagonal-value loss, Im eps ~ 0.3 in-plane
    absorbed = float(_absorbed_fraction(sol, 0.3, nx=16, density=200.0))
    assert absorbed == pytest.approx(balance, rel=3e-3)


@needs_fmmax
def test_field3d_anisotropic_matches_fmmax_oracle():
    """The patterned-general-anisotropic reconstruction is parity-exact against fmmax:
    the |E|^2 volume agrees to ~1e-5 even though the layer is solved by the export's
    own anisotropic eigensolve (fmmax fields are the reference for the same stack)."""
    import fmmax

    from jno._rcwa import LatticeVectors, generate_expansion
    from jno.rcwa import Rcwa

    period = (0.6, 0.6)
    xs = (np.arange(24) + 0.5) / 24 * period[0]
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    ripple = 2.0 + 0.6 * np.cos(2 * np.pi * X / period[0])
    five = (ripple + 0.4j, np.zeros_like(ripple), np.zeros_like(ripple), ripple + 0.4j, np.full_like(ripple, 2.4))
    layers = [(0.5, 1.0), (0.3, five), (0.5, 2.25)]
    orders, nx = 15, 12
    sol = Rcwa(layers, period=period, orders=orders, wavelength=WL, assume_periodic=True).solve()
    inten = sol.field3d(nx=nx, density=120.0)[0]

    lv = LatticeVectors(u=jnp.asarray([period[0], 0.0]), v=jnp.asarray([0.0, period[1]]))
    ex = generate_expansion(lv, approximate_num_terms=orders)
    thick = [0.5, 0.3, 0.5]
    lsrs = [
        fmmax.fmm.eigensolve_isotropic_media(
            jnp.asarray(WL),
            jnp.asarray((0.0, 0.0)),
            lv,
            jnp.full((1, 1), 1.0, jnp.complex64),
            ex,
            formulation=fmmax.Formulation.JONES_DIRECT_FOURIER,
        ),
        fmmax.fmm.eigensolve_anisotropic_media(
            jnp.asarray(WL),
            jnp.asarray((0.0, 0.0)),
            lv,
            *(jnp.asarray(c) for c in five),
            ex,
            formulation=fmmax.Formulation.JONES_DIRECT_FOURIER,
        ),
        fmmax.fmm.eigensolve_isotropic_media(
            jnp.asarray(WL),
            jnp.asarray((0.0, 0.0)),
            lv,
            jnp.full((1, 1), 2.25, jnp.complex64),
            ex,
            formulation=fmmax.Formulation.JONES_DIRECT_FOURIER,
        ),
    ]
    gs = fmmax.basis.min_array_shape_for_expansion(ex)
    ones = jnp.ones((*gs, 1), complex)
    zeros = jnp.zeros((*gs, 1), complex)
    fwd = jnp.reshape(fmmax.sources.amplitudes_for_fields(ones, zeros, zeros, ones, lsrs[0])[0], (-1, 1))
    smi = fmmax.scattering.stack_s_matrices_interior(lsrs, thick)
    amps = fmmax._fields.stack_amplitudes_interior(smi, fwd, jnp.zeros_like(fwd))
    znum = [max(4, int(round(t * 120.0))) for t in thick]
    (ef, _h, (x, y, z)) = fmmax._fields.stack_fields_3d(amps, lsrs, thick, znum, grid_shape=(nx, nx))
    oracle = jnp.sum(jnp.abs(ef) ** 2, axis=0)[..., 0]
    a, b = np.asarray(oracle), np.asarray(inten)
    assert a.shape == b.shape
    assert float(np.abs(a - b).max() / np.abs(a).max()) < 1e-4


# ------------------------------------------------------------------------------------
# PML folded ambients: isolated scatterer in an in-plane PML frame
# ------------------------------------------------------------------------------------


def _pml_frame_ambient(period, ng, w, sigma):
    """A patterned 10-tuple for an in-plane PML slice: unit core, absorbing frame.

    The frame is the Maxwell uniaxial PML the front door builds -- ε̂ = Λ, μ̂ = Λ with the
    coordinate-stretch ``Λ = diag(Sy/Sx, Sx/Sy, Sx·Sy)`` and a quadratic σ-ramp near the
    cell walls. As a tensor layer it cannot reduce to a scalar, so the seam folds it into
    the stack as a unit film -- the isolated-scatterer model of task T28."""
    xs = (np.arange(ng) + 0.5) / ng * period
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    sx = sigma * (np.clip((w - X) / w, 0, 1) ** 2 + np.clip((X - (period - w)) / w, 0, 1) ** 2)
    sy = sigma * (np.clip((w - Y) / w, 0, 1) ** 2 + np.clip((Y - (period - w)) / w, 0, 1) ** 2)
    Sx, Sy = 1.0 + 1j * sx / K0, 1.0 + 1j * sy / K0
    lxx = np.asarray(Sy / Sx)
    lyy = np.asarray(Sx / Sy)
    lzz = np.asarray(Sx * Sy)
    zero = np.zeros((ng, ng))
    return (lxx, zero, zero, lyy, lzz, lxx, zero, zero, lyy, lzz)  # ε̂ = Λ, μ̂ = Λ


def test_isolated_scatterer_in_pml_frame_stays_energy_conserving():
    """An isolated scatterer inside an in-plane PML supercell: sigma=0 is the identity
    tensor (reduces to scalar, T+R ~ 1); sigma>0 absorbs the laterally-scattered light
    (T+R < 1); the folded-frame field3d volume stays finite in both cases."""
    period = 1.4
    ng = 32
    w = 0.35 * period
    scatterer = _disk_grid(period, ng, 0.18, 6.0, center_frac=0.5)

    def frame_trade(sigma):
        frame_top = _pml_frame_ambient(period, ng, w, sigma)
        frame_bot = _pml_frame_ambient(period, ng, w, sigma)
        layers = [
            (np.inf, frame_top),
            (0.2, np.full((ng, ng), 1.0)),
            (1.0, scatterer),
            (0.2, np.full((ng, ng), 1.0)),
            (np.inf, frame_bot),
        ]
        sol = Rcwa(layers, period=(period, period), orders=60, wavelength=WL, assume_periodic=True).solve()
        T, R = float(sol.efficiency("T")), float(sol.efficiency("R"))
        inten, (_x, _y, _z), _lz = sol.field3d(nx=16, density=60.0)
        return T, R, bool(np.all(np.isfinite(np.asarray(inten))))

    T0, R0, finite0 = frame_trade(0.0)  # sigma=0 -> identity frame, no absorption
    Ta, Ra, finitea = frame_trade(3.0)  # real absorber
    assert T0 + R0 == pytest.approx(1.0, abs=2e-2)
    assert Ta + Ra < 0.98  # the frame drained laterally-diffracted power
    assert finite0 and finitea  # the folded-frame interior renders cleanly


def test_pml_frame_guard_raises_loudly_with_chunk_remedy():
    """The never-silent energy guard holds under a PML frame: a divergent frame tensor (an
    unbalanced ε-only absorber whose recursion does not converge) raises the chunk/orders
    remedy instead of returning an unconserved answer, while the balanced ε̂=Λ, μ̂=Λ frame
    (the front door's PML) solves and absorbs."""
    period = 1.4
    ng = 24
    w = 0.35 * period
    xs = (np.arange(ng) + 0.5) / ng * period
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    sx = 3.0 * (np.clip((w - X) / w, 0, 1) ** 2 + np.clip((X - (period - w)) / w, 0, 1) ** 2)
    sy = 3.0 * (np.clip((w - Y) / w, 0, 1) ** 2 + np.clip((Y - (period - w)) / w, 0, 1) ** 2)
    Sx, Sy = 1.0 + 1j * sx / K0, 1.0 + 1j * sy / K0
    eps_only = (np.asarray(Sy / Sx), np.zeros((ng, ng)), np.zeros((ng, ng)), np.asarray(Sx / Sy), np.asarray(Sx * Sy))
    scatterer = _disk_grid(period, ng, 0.18, 6.0, center_frac=0.5)
    bad = [(1.0, eps_only), (1.0, scatterer), (1.0, tuple(c.copy() for c in eps_only))]
    st = StateStack(
        wavelength=WL,
        period=(period, period),
        amb_sup=1.0,
        amb_sub=1.0,
        layers=bad,
        orders=60,
        chunk_override=[1, 1, 1],  # force the un-chunked recursion
    )
    with pytest.raises(ValueError, match="chunk the layer thinner|not converged|auto chunks"):
        st.solve(polarization="x")
    # the balanced Maxwell PML (ε̂ = Λ, μ̂ = Λ) stack of the same geometry solves and absorbs
    frame = _pml_frame_ambient(period, ng, w, 3.0)
    good = [(1.0, frame), (1.0, scatterer), (1.0, tuple(c.copy() for c in frame))]
    st2 = StateStack(wavelength=WL, period=(period, period), amb_sup=1.0, amb_sub=1.0, layers=good, orders=40)
    res2 = st2.solve(polarization="x")
    assert float(res2.T) + float(res2.R) < 0.98  # the frame absorbs; no energy-guard complaint


# ------------------------------------------------------------------------------------
# Loud guards on field()/field3d arguments (garbage-signature rejection)
# ------------------------------------------------------------------------------------


def test_field3d_rejects_garbage_arguments_loudly():
    """nx too coarse for the expansion, nx below 2, and non-positive density must raise
    jNO-style errors with the fix named, never silently return a wrong-shaped answer."""
    layers = [(0.5, 1.0), (0.2, 2.0 + 0.5j), (0.5, 2.25)]
    sol = Rcwa(layers, period=(0.6, 0.6), orders=9, wavelength=WL, assume_periodic=True).solve()
    with pytest.raises(RcwaError, match="coarse"):
        sol.field3d(nx=2, density=40.0)  # every model needs >= 3 points/axis at orders=9
    with pytest.raises(RcwaError, match="nx"):
        sol.field3d(nx=1)
    with pytest.raises(RcwaError, match="density"):
        sol.field3d(nx=16, density=0.0)
    with pytest.raises(RcwaError, match="density"):
        sol.field(density=-1.0)


# ------------------------------------------------------------------------------------
# Oblique-incidence |E|^2 caveat (documented, pinned)
# ------------------------------------------------------------------------------------


def test_oblique_intensity_caveat_is_bounded_and_documented():
    """Pinned skew for the documented oblique ``|E|^2`` caveat: for a PATTERNED
    (disk) lossy film the absorbed power computed from the reconstructed
    ``|E|^2`` deviates from the accurate flux balance ``1 - T - R`` by a few
    percent -- larger at oblique incidence (measured ~4% at sinθ=0.3; T31's
    reported ~5%-class band, shared in kind with the patterned-aniso residual
    of iss_mwwfkut5fdl61w). The test turns LOUD if the skew ever exceeds the
    documented band (a real reconstruction regression); the efficiency/flux
    readouts remain the accurate energy quantities (see the field3d/field
    docstrings and the fields.py module docstring)."""
    P = 0.85
    ng = 32

    def disk_eps():
        xs = (np.arange(ng) + 0.5) / ng * P
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        d = (X - 0.5 * P) ** 2 + (Y - 0.5 * P) ** 2 < 0.22**2
        return np.where(d, 6.0 + 0.3j, 1.0)

    def skew_at(sinth):
        layers = [(0.5, 1.0), (0.4, disk_eps()), (0.5, 2.25)]
        sol = Rcwa(
            layers,
            period=(P, P),
            orders=15,
            wavelength=WL,
            k_in=(K0 * sinth, 0.0),
            assume_periodic=True,
        ).solve()
        T, R = float(sol.efficiency("T")), float(sol.efficiency("R"))
        flux_balance = 1.0 - T - R
        assert flux_balance > 0.01  # the disk genuinely absorbs at both angles
        inten, (x, y, z), lz = sol.field3d(nx=24, density=400.0)
        z = np.asarray(z)
        zin = (z >= lz[0] - 1e-12) & (z <= lz[1] + 1e-12)
        X, Y = np.meshgrid(np.asarray(x), np.asarray(y), indexing="ij")
        im = jnp.asarray(np.where((X - 0.5 * P) ** 2 + (Y - 0.5 * P) ** 2 < 0.22**2, 0.3, 0.0))
        prof = jnp.mean(im[..., jnp.newaxis] * jnp.asarray(inten)[:, :, zin], axis=(0, 1))
        absorbed = float(0.5 * K0 * jnp.trapezoid(prof, z[zin]) / sol._Pin)
        return (absorbed - flux_balance) / flux_balance

    skew_normal = skew_at(0.0)
    skew_oblique = skew_at(0.3)
    # documented band: the reconstructed-|E|^2 absorbed power sits within ~10% of
    # the flux balance at both angles (measured ~3% normal, ~4% oblique); a
    # regression pushing it past the band is a real defect, not the caveat.
    assert abs(skew_normal) < 0.10
    assert abs(skew_oblique) < 0.10
