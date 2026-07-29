"""jno.rcwa aerial imaging — sol.aerial(): the partially-coherent litho image on top of the rigorous mask.

The RCWA solve gives the mask's diffraction spectrum; ``sol.aerial(NA, source, …)`` projects it through a
lens (NA + pupil) and sums over the illumination (Abbe) to form the wafer-plane intensity. Validated against
the imaging limits (open frame → uniform; partial coherence reduces contrast) and differentiable in the mask
design (ILT) — jax.grad of an image functional matches finite difference.
"""

import importlib.util
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # the fmmax solve OOMs on a small GPU at these orders

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jno  # noqa: E402
from jno.trace import FunctionCall  # noqa: E402
from jno.trace_evaluator import TraceEvaluator  # noqa: E402
from jno.utils.solver.fem_adapt import _domain_from_arrays  # noqa: E402

HAS_FMMAX = importlib.util.find_spec("fmmax") is not None
needs_fmmax = pytest.mark.skipif(not HAS_FMMAX, reason="fmmax (jno.rcwa backend) not installed")
pytest.importorskip("pygmsh", reason="pygmsh required for box meshing")

K0 = 2 * jnp.pi
LZ = 3.0
P = 2.8  # mask period (non-integer × wavelength -> avoids a Rayleigh anomaly)


def _sbox(dx=0.07, ny=5):
    xs = np.linspace(0, P, int(round(P / dx)) + 1)
    ys = np.linspace(0, P, ny)
    zs = np.linspace(0, LZ, int(round(LZ / 0.2)) + 1)
    nx, nyy, nz = len(xs), len(ys), len(zs)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Pt = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    vid = lambda i, j, k: (i * nyy + j) * nz + k  # noqa: E731
    CUBE = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    TE = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
    tets = []
    for i in range(nx - 1):
        for j in range(nyy - 1):
            for k in range(nz - 1):
                c = [vid(i + a, j + b, k + cc) for (a, b, cc) in CUBE]
                tets += [[c[t[0]], c[t[1]], c[t[2]], c[t[3]]] for t in TE]
    tets = np.asarray(tets)
    F = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    uq, cnt = np.unique(np.sort(F, 1), axis=0, return_counts=True)
    d = _domain_from_arrays(jno.Shape.box(0, 0, 0, P, P, LZ, size=0.2).domain(), Pt, tets, uq[cnt == 1], copy=True)
    e = 1e-6
    for nm, f in [
        ("bottom", lambda x, y, z: z < e),
        ("top", lambda x, y, z: z > LZ - e),
        ("left", lambda x, y, z: x < e),
        ("right", lambda x, y, z: x > P - e),
        ("front", lambda x, y, z: y < e),
        ("back", lambda x, y, z: y > P - e),
    ]:
        d.tag(nm, f)
    return d


def _cons(eps_fn, amp=None):
    d = _sbox()
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)

    def fc(n):
        c = d.variable(n, split=True)
        return u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])

    ubt, vbt = fc("bottom")
    utp, vtp = fc("top")
    ul, _ = fc("left")
    ur, _ = fc("right")
    uf, _ = fc("front")
    ub, _ = fc("back")
    eps = jno.fn(eps_fn, [xi, yi, zi])
    return [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * utp) * vtp,
        -(1j * K0 * ubt - 2j * K0) * vbt,
        ul - ur,
        uf - ub,
    ]


def _line(x, y, z):  # a 1-D line/space grating (lines in x, uniform in y), eps=2 lines
    return jnp.where((jnp.abs(x - P / 2) < P * 0.28) & (z >= 1.0) & (z < 1.2), 2.0, 1.0)


def _open(x, y, z):  # open frame (uniform vacuum -> a clear mask)
    return 1.0 + 0.0 * z


def _contrast(img):
    prof = np.asarray(img).mean(1)  # average over y (grating uniform in y)
    return (prof.max() - prof.min()) / (prof.max() + prof.min())


@needs_fmmax
def test_open_frame_gives_uniform_image():
    """A clear (open-frame) mask images to a UNIFORM aerial image — no structure to diffract."""
    img = np.asarray(jno.rcwa(_cons(_open), orders=40, grid=48).solve().aerial(NA=0.6, source=0.5))
    assert (img.max() - img.min()) / (img.mean() + 1e-12) < 1e-3


@needs_fmmax
def test_partial_coherence_reduces_contrast():
    """Increasing the partial-coherence factor σ (a larger conventional source) reduces the image contrast —
    the defining behaviour of partially-coherent imaging."""
    sol = jno.rcwa(_cons(_line), orders=60, grid=56).solve()
    c_lo = _contrast(sol.aerial(NA=0.6, source=0.2))
    c_hi = _contrast(sol.aerial(NA=0.6, source=0.9))
    assert c_lo > c_hi > 0.0  # contrast drops with σ, but the grating is still resolved


@needs_fmmax
def test_source_shapes_run():
    """Conventional (float), annular (tuple), and a raw pupil-weight array all form a valid image."""
    sol = jno.rcwa(_cons(_line), orders=40, grid=48).solve()
    conv = sol.aerial(NA=0.6, source=0.5)
    ann = sol.aerial(NA=0.6, source=(0.4, 0.8))
    arr = sol.aerial(NA=0.6, source=np.ones((15, 15)))  # freeform pupil
    for im in (conv, ann, arr):
        im = np.asarray(im)
        assert im.shape == (128, 128) and np.all(np.isfinite(im)) and im.min() >= 0.0


def _nrm(img):  # image normalised to unit mean, to compare shape/contrast independent of total energy
    a = np.asarray(img)
    return a / (a.mean() + 1e-12)


@needs_fmmax
def test_vector_reduces_to_scalar_as_na_drops():
    """The vector (high-NA) image must reduce to the scalar one as NA→0: the pupil becomes the identity and
    E_z→0. So the (energy-normalised) x-polarised vector image departs from the scalar image LESS at low NA
    than at high NA -- the defining consistency check of the Richards-Wolf/Flagello vector model."""
    sol = jno.rcwa(_cons(_line), orders=60, grid=56).solve()

    def dev(NA):  # normalised L2 departure of the vector-x image from the scalar image
        sc, vx = _nrm(sol.aerial(NA=NA, source=0.3)), _nrm(sol.aerial(NA=NA, source=0.3, polarization="x"))
        return float(np.linalg.norm(vx - sc) / np.linalg.norm(sc))

    assert dev(0.4) < dev(0.75)  # closer to scalar at lower NA (smaller ray angles)


@needs_fmmax
def test_vector_tm_te_contrast_split():
    """The vector signature at high NA: for a grating periodic in x (orders spread in the x–z plane), x-pol is
    TM (E tilts with the ray → the interfering beams lose projection → lower contrast) and y-pol is TE (E stays
    parallel → full contrast). So TE contrast must exceed TM contrast at high NA."""
    sol = jno.rcwa(_cons(_line), orders=60, grid=56).solve()
    c_tm = _contrast(sol.aerial(NA=0.9, source=0.3, polarization="x"))  # TM (E in plane of incidence)
    c_te = _contrast(sol.aerial(NA=0.9, source=0.3, polarization="y"))  # TE (E out of plane)
    assert c_te > c_tm > 0.0


@needs_fmmax
def test_vector_pol_shapes_run():
    """'x', 'y' and 'unpolarized' all form a valid image, and the unpolarized image is exactly the mean of the
    two linear-polarization images (incoherent averaging)."""
    sol = jno.rcwa(_cons(_line), orders=40, grid=48).solve()
    ix = np.asarray(sol.aerial(NA=0.7, source=0.5, polarization="x"))
    iy = np.asarray(sol.aerial(NA=0.7, source=0.5, polarization="y"))
    iu = np.asarray(sol.aerial(NA=0.7, source=0.5, polarization="unpolarized"))
    for im in (ix, iy, iu):
        assert im.shape == (128, 128) and np.all(np.isfinite(im)) and im.min() >= 0.0
    assert np.allclose(iu, 0.5 * (ix + iy), rtol=1e-6, atol=1e-9)


def _ilt_cons():
    """A mask parameterised by the line permittivity `ep` (a jno.np.parameter), for the inverse-lithography
    gradient tests. Returns (constraint list, the ep parameter)."""
    d = _sbox()
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)

    def fc(n):
        c = d.variable(n, split=True)
        return u.bind(x=c[0], y=c[1], z=c[2]), phi.bind(x=c[0], y=c[1], z=c[2])

    ubt, vbt = fc("bottom")
    utp, vtp = fc("top")
    ul, _ = fc("left")
    ur, _ = fc("right")
    uf, _ = fc("front")
    ub, _ = fc("back")
    ep = jno.np.parameter((), name="ep").initialize(jax.nn.initializers.constant(2.0))
    ind = jno.fn(
        lambda x, y, z: jnp.where((jnp.abs(x - P / 2) < P * 0.28) & (z >= 1.0) & (z < 1.2), 1.0, 0.0), [xi, yi, zi]
    )
    eps = 1.0 + ind * (ep - 1.0)
    cons = [
        ui.x * vi.x + ui.y * vi.y + ui.z * vi.z - K0**2 * eps * (u * vi),
        -(1j * K0 * utp) * vtp,
        -(1j * K0 * ubt - 2j * K0) * vbt,
        ul - ur,
        uf - ub,
    ]
    return cons, ep


def _grad_vs_fd(node, ep):
    """jax.grad of mean(node²) w.r.t. the ep parameter, and the central finite difference, at ep=2."""

    def loss(v):
        mod = eqx.tree_at(lambda m: m.value, ep.model.module, jnp.asarray(v))
        return jnp.mean(TraceEvaluator({ep.model.layer_id: mod}).evaluate(node) ** 2)

    h = 1e-2
    return float(jax.grad(loss)(2.0)), (float(loss(2.0 + h)) - float(loss(2.0 - h))) / (2 * h)


@needs_fmmax
def test_aerial_is_differentiable_ilt():
    """The aerial image is differentiable in the mask design (inverse lithography): rc.solve().aerial() is a
    trace node, and jax.grad of an image functional w.r.t. a mask parameter matches a finite difference —
    the gradient flows through the imaging AND the RCWA mask solve."""
    cons, ep = _ilt_cons()
    node = jno.rcwa(cons, orders=60, grid=56, params={"ep": 2.0}).solve().aerial(NA=0.6, source=0.5)
    assert isinstance(node, FunctionCall)
    g, fd = _grad_vs_fd(node, ep)
    assert g == pytest.approx(fd, rel=3e-3)
    assert abs(g) > 1e-3


def _linewidth(img):  # fraction of the (y-averaged) x-profile that prints as a feature (> 0.5)
    return float((np.asarray(img).mean(1) > 0.5).mean())


def _aerial_band(sol, **kw):  # the (min, max, span) of this mask's aerial intensity — a weak dielectric
    a = np.asarray(sol.aerial(**kw))  # grating modulates around ~1, so a threshold must sit INSIDE this band
    lo, hi = float(a.min()), float(a.max())
    return lo, hi, max(hi - lo, 1e-9)


def _aerial_band_exp(exp):  # same band, read from an exposure object's intensity
    a = np.asarray(exp.intensity())
    lo, hi = float(a.min()), float(a.max())
    return lo, hi, max(hi - lo, 1e-9)


@needs_fmmax
def test_develop_matches_printed_shortcut():
    """The exposure/develop seam and the printed() shortcut are the same computation: exposing then
    developing with a Threshold resist equals printed(resist=that Threshold)."""
    sol = jno.rcwa(_cons(_line), orders=40, grid=48).solve()
    r = jno.litho.Threshold(threshold=0.3, diffusion=0.02, steepness=40.0)
    exp = sol.expose(NA=0.6, source=0.4)
    via_develop = np.asarray(exp.develop(r))
    via_printed = np.asarray(sol.printed(NA=0.6, source=0.4, resist=r))
    assert np.allclose(via_develop, via_printed, rtol=1e-10, atol=1e-12)
    assert np.allclose(np.asarray(exp.intensity()), np.asarray(sol.aerial(NA=0.6, source=0.4)))  # optics = aerial


@needs_fmmax
def test_printed_is_in_range_and_structured():
    """The developed resist image (via a Threshold resist) is a soft mask in [0, 1] and thresholds the line
    grating into a printed feature — both cleared (≈0) and remaining (≈1) resist are present."""
    sol = jno.rcwa(_cons(_line), orders=60, grid=56).solve()
    lo, hi, span = _aerial_band(sol, NA=0.6, source=0.4)
    r = jno.litho.Threshold(threshold=0.5 * (lo + hi), steepness=24.0 / span)
    img = np.asarray(sol.expose(NA=0.6, source=0.4).develop(r))
    assert img.min() >= 0.0 and img.max() <= 1.0 and np.all(np.isfinite(img))
    assert img.min() < 0.05 and img.max() > 0.95  # a real bilevel pattern, not a flat grey


@needs_fmmax
def test_threshold_sets_linewidth():
    """Raising the dose-to-clear threshold shrinks the printed feature — the CD knob is monotone."""
    exp = jno.rcwa(_cons(_line), orders=60, grid=56).solve().expose(NA=0.6, source=0.4)
    lo, _hi, span = _aerial_band_exp(exp)
    k = 24.0 / span
    w_lo = _linewidth(exp.develop(jno.litho.Threshold(threshold=lo + 0.35 * span, steepness=k)))
    w_hi = _linewidth(exp.develop(jno.litho.Threshold(threshold=lo + 0.65 * span, steepness=k)))
    assert w_lo > w_hi  # higher threshold -> narrower printed feature


@needs_fmmax
def test_peb_diffusion_smooths():
    """A larger PEB diffusion length blurs the aerial image before development, so its modulation shrinks and
    the printed pattern loses contrast (max−min) — the defining effect of acid diffusion."""

    def contrast(img):
        a = np.asarray(img)
        return float(a.max() - a.min())

    exp = jno.rcwa(_cons(_line), orders=60, grid=56).solve().expose(NA=0.6, source=0.4)
    lo, hi, span = _aerial_band_exp(exp)
    thr, k = 0.5 * (lo + hi), 12.0 / span
    c0 = contrast(exp.develop(jno.litho.Threshold(threshold=thr, steepness=k, diffusion=0.0)))
    c1 = contrast(exp.develop(jno.litho.Threshold(threshold=thr, steepness=k, diffusion=1.0)))
    assert c1 < c0


@needs_fmmax
def test_printed_is_differentiable_ilt():
    """The full computational-lithography chain is differentiable: jax.grad of a printed-resist functional
    w.r.t. the mask permittivity matches finite difference — the gradient flows through development, imaging
    AND the RCWA mask solve (the point of the resist model: ILT/SMO with the resist in the loop)."""
    cons, ep = _ilt_cons()
    rc = jno.rcwa(cons, orders=60, grid=56, params={"ep": 2.0})
    lo, hi, span = _aerial_band(rc.solve(params={"ep": 2.0}), NA=0.6, source=0.5)  # band at the eval point
    thr, k = 0.5 * (lo + hi), 6.0 / span  # threshold in-band + moderate contrast -> the sigmoid stays active
    resist = jno.litho.Threshold(threshold=thr, diffusion=0.1, steepness=k)
    node = rc.solve().printed(NA=0.6, source=0.5, resist=resist)
    assert isinstance(node, FunctionCall)
    g, fd = _grad_vs_fd(node, ep)
    assert g == pytest.approx(fd, rel=5e-3)
    assert abs(g) > 1e-3


@needs_fmmax
def test_vector_printed_is_differentiable_ilt():
    """**Vector high-NA** imaging is differentiable inside the parametric ILT node: jax.grad of an
    unpolarized (TE+TM) printed-resist functional w.r.t. the mask permittivity matches finite difference —
    so vector high-NA ILT/SMO trains, not just the scalar model. Guards the vector-pupil sqrt(0) safe-guard
    (the gradient is NaN without it) and, at a converged truncation, the (hypersensitive) y-polarization path."""
    cons, ep = _ilt_cons()
    rc = jno.rcwa(cons, orders=60, grid=56, params={"ep": 2.0})
    lo, hi, span = _aerial_band(rc.solve(params={"ep": 2.0}), NA=0.6, source=0.5, polarization="unpolarized")
    resist = jno.litho.Threshold(threshold=0.5 * (lo + hi), diffusion=0.1, steepness=6.0 / span)
    node = rc.solve().printed(NA=0.6, source=0.5, polarization="unpolarized", resist=resist)
    assert isinstance(node, FunctionCall)
    g, fd = _grad_vs_fd(node, ep)
    assert np.isfinite(g)  # sqrt(0) in the vector pupil makes this NaN without the safe-sqrt guard
    assert g == pytest.approx(fd, rel=1e-2)
    assert abs(g) > 1e-3


def test_caresist_validation():
    """CAResist rejects a bad tone or a wrong-length rate-constant tuple (cheap — no solve)."""
    with pytest.raises(ValueError):
        jno.litho.CAResist(tone="sideways")
    with pytest.raises(ValueError):
        jno.litho.CAResist(k=(0.5, 0.005, 0.005))  # needs (k1..k5)


class _FakeAerial:  # a stub exposure carrying a prescribed aerial image, for the 2-D PEB (no RCWA solve)
    def __init__(self, img, period):
        self._img, self.period = img, period

    def intensity(self):
        return self._img


def test_caresist_develops_exposed_band():
    """The rigorous 2-D reaction-diffusion PEB develops a pattern that TRACKS the exposure: seeded (Dill acid)
    from a bright band, the exposed band deprotects and clears while the dark edges stay -- checked by a strong
    positive correlation between the developed field and the exposure (clears where bright), not just a coarse
    profile. The dose is calibrated so the latent acid straddles the quencher (a saturated or starved seed
    develops flat -- the honest operating point). This is the litho-level regression test for the periodic-tie
    prolongation fix: the pre-fix reduced-vs-full DOF slice scrambled the developed pattern to ~zero
    correlation (a coarse mid>edge check still passed on the scramble, which is why the bug hid here)."""
    G = 40
    xs = (np.arange(G) + 0.5) / G
    band = np.where((xs > 0.35) & (xs < 0.65), 1.0, 0.05)  # bright exposed band in x, uniform in y
    img = jnp.asarray(band[:, None] * np.ones((1, G)))
    resist = jno.litho.CAResist(n=24, t_peb=30.0, steps=16, dill_c=1.0, dose=2.0, diffusion_length=(0.03, 0.03))
    dev = np.asarray(resist(_FakeAerial(img, (1.2, 1.2))))
    assert dev.shape == (24, 24) and np.all(np.isfinite(dev))
    assert dev.min() >= 0.0 and dev.max() <= 1.0  # developed fraction clipped to the physical [0, 1]
    idx = np.linspace(0, G - 1, 24).astype(int)  # sample the exposure onto the developed grid
    expo = np.asarray(img)[np.ix_(idx, idx)]
    corr = float(np.corrcoef(dev.ravel(), expo.ravel())[0, 1])
    assert corr > 0.6, f"developed pattern must track the exposure (clears where bright), corr={corr:.2f}"
    prof, nn = dev.mean(1), dev.shape[0]  # x-profile (band uniform in y)
    mid = prof[nn // 4 : 3 * nn // 4].mean()  # the exposed band (centre in x)
    edge = np.concatenate([prof[: nn // 4], prof[3 * nn // 4 :]]).mean()
    assert mid > edge + 0.2  # exposed band clears clearly more (strong, not the 0.02 the DOF scramble faked)


@needs_fmmax
def test_caresist_is_differentiable_ilt():
    """ILT through the RIGOROUS resist: jax.grad of a developed-pattern loss w.r.t. the mask permittivity
    matches finite difference through the WHOLE chain — RCWA mask solve → aerial image → Dill latent acid →
    transient reaction-diffusion PEB → developed M. This end-to-end gradient (design the mask against the
    physically-baked pattern) is exactly what a forward-only litho simulator cannot provide."""
    cons, ep = _ilt_cons()
    resist = jno.litho.CAResist(n=12, t_peb=15.0, steps=4, dill_c=4.0, diffusion_length=(0.12, 0.08))
    node = jno.rcwa(cons, orders=30, grid=24, params={"ep": 2.0}).solve().printed(NA=0.6, source=0.5, resist=resist)
    assert isinstance(node, FunctionCall)
    g, fd = _grad_vs_fd(node, ep)
    assert g == pytest.approx(fd, rel=5e-3)
    assert abs(g) > 1e-4


@needs_fmmax
def test_bulk_reduces_to_aerial_at_top():
    """The standing-wave bulk image reduces to the aerial image at the top of a vacuum film (z=0, no substrate
    reflection): exposure.bulk(vacuum)[:, :, 0] == exposure.intensity()."""
    exp = jno.rcwa(_cons(_line), orders=40, grid=40).solve().expose(NA=0.6, source=0.4)
    aer = np.asarray(exp.intensity())
    vol = np.asarray(exp.bulk(jno.litho.Film(n_resist=1.0, thickness=1.0, n_substrate=1.0, nz=6)))
    assert vol.shape == (aer.shape[0], aer.shape[1], 6)
    assert np.allclose(vol[:, :, 0], aer, rtol=1e-6, atol=1e-9)


@needs_fmmax
def test_bulk_standing_wave_period():
    """A reflective substrate makes each order interfere with its reflection → a vertical STANDING WAVE of
    period λ/(2·n_resist). Open frame (single order), n_resist=1.7, wl=1 (K0=2π) → period ≈ 0.294."""
    nr, d, nz = 1.7, 2.0, 64
    # The measurement is one pixel column FFT'd along z, and the mask is an open frame (laterally
    # uniform), so lateral resolution buys nothing here -- while the default 128x128 image makes the
    # Abbe sum's (n_source, nz, grid, grid) stack a multi-GB allocation. Keep nz (it sets the FFT
    # resolution the period is read from) and drop the image to 16x16.
    exp = jno.rcwa(_cons(_open), orders=20, grid=24).solve().expose(NA=0.4, source=0.3, grid=16)
    vol = np.asarray(exp.bulk(jno.litho.Film(n_resist=nr, thickness=d, n_substrate=4.0, nz=nz)))
    Iz = vol[vol.shape[0] // 2, vol.shape[1] // 2, :]  # I(z) at a pixel
    Iz = Iz - Iz.mean()
    period = 1.0 / np.fft.rfftfreq(nz, d=d / (nz - 1))[1:][np.argmax(np.abs(np.fft.rfft(Iz))[1:])]
    assert period == pytest.approx(1.0 / (2 * nr), rel=0.15)


@needs_fmmax
def test_bulk_absorption_decays():
    """An absorbing resist (complex n_resist), index-matched to the substrate (no reflection), attenuates the
    field with depth: the depth-averaged intensity at the film bottom is below the top."""
    exp = jno.rcwa(_cons(_open), orders=20, grid=24).solve().expose(NA=0.4, source=0.3)
    vol = np.asarray(exp.bulk(jno.litho.Film(n_resist=1.7 + 0.15j, thickness=2.0, n_substrate=1.7 + 0.15j, nz=12)))
    depth_mean = vol.mean((0, 1))  # mean over (x, y) at each depth
    assert depth_mean[-1] < depth_mean[0]  # absorbed with depth


class _FakeExposure:  # a stub exposure carrying a prescribed bulk image, for testing the 3-D PEB directly
    def __init__(self, bulk, period):
        self._bulk, self.period = bulk, period

    def bulk(self, film, source_chunk=None):  # mirror _Exposure.bulk -- CAResist forwards source_chunk
        return self._bulk


def test_caresist_3d_develops_exposed_stripe():
    """The 3-D CAResist (a jno.Shape box, periodic in x,y, species diffusing in x,y AND z) develops a printed
    pattern that TRACKS the exposure: seeded from a bulk image with a bright exposed stripe in x, the exposed
    region deprotects and clears while the dark edges stay -- a strong positive correlation with the stripe
    (the periodic-tie prolongation must be right for the developed volume to follow the seed; the pre-fix
    reduced-vs-full DOF slice scrambled it). A stub exposure gives unambiguous contrast -- fast, no RCWA."""
    G, nz = 24, 6
    xs = (np.arange(G) + 0.5) / G
    band = np.where((xs > 0.3) & (xs < 0.7), 1.0, 0.05)  # bright exposed stripe in x, uniform in y and z
    bulk = jnp.asarray(band[:, None, None] * np.ones((1, G, nz)))
    film = jno.litho.Film(n_resist=1.6, thickness=0.6, nz=nz)
    resist = jno.litho.CAResist(n=16, t_peb=20.0, steps=6, dill_c=4.0, diffusion_length=(0.12, 0.08), film=film)
    dev = np.asarray(resist(_FakeExposure(bulk, (1.2, 1.2))))
    assert dev.shape == (16, 16, nz) and np.all(np.isfinite(dev))
    assert dev.min() >= 0.0 and dev.max() <= 1.0  # developed fraction clipped to the physical [0, 1]
    n = dev.shape[0]  # the developed volume tracks the stripe (clears where bright), through the whole depth
    bn = np.where(((np.arange(n) + 0.5) / n > 0.3) & ((np.arange(n) + 0.5) / n < 0.7), 1.0, 0.05)
    stripe = bn[:, None, None] * np.ones((1, n, dev.shape[2]))
    assert float(np.corrcoef(dev.ravel(), stripe.ravel())[0, 1]) > 0.6
    prof, nn = dev.mean((1, 2)), dev.shape[0]  # x-profile
    mid = prof[nn // 4 : 3 * nn // 4].mean()  # exposed stripe (centre in x)
    edge = np.concatenate([prof[: nn // 4], prof[3 * nn // 4 :]]).mean()
    assert mid > edge + 0.3  # exposed stripe clears clearly more (strong, not the 0.02 the DOF scramble faked)


@needs_fmmax
def test_caresist_3d_end_to_end():
    """The 3-D PEB wires end to end through the real optics: exp.develop(CAResist(film=...)) reads the
    standing-wave bulk image, solves the 3-D reaction-diffusion PEB on a jno.Shape box (periodic x,y via a
    conforming remesh), and returns a finite developed (n, n, nz) volume in [0, 1]."""
    exp = jno.rcwa(_cons(_line), orders=40, grid=40).solve().expose(NA=0.6, source=0.4)
    film = jno.litho.Film(n_resist=1.6, thickness=0.6, n_substrate=4.0, nz=6)
    resist = jno.litho.CAResist(n=14, t_peb=15.0, steps=4, dill_c=4.0, diffusion_length=(0.12, 0.08), film=film)
    dev = np.asarray(exp.develop(resist))
    assert dev.shape == (14, 14, 6) and np.all(np.isfinite(dev))
    assert dev.min() >= -1e-3 and dev.max() <= 1.0 + 1e-3


@needs_fmmax
def test_caresist_3d_is_differentiable_ilt():
    """ILT through the 3-D PEB: jax.grad of a developed-volume loss w.r.t. the mask permittivity matches
    finite difference through the WHOLE chain — RCWA solve → standing-wave bulk image → 3-D reaction-diffusion
    PEB on a jno.Shape box → developed volume. The rigorous depth-resolved resist stays fully differentiable."""
    cons, ep = _ilt_cons()
    film = jno.litho.Film(n_resist=1.6, thickness=0.6, n_substrate=4.0, nz=4)
    # Sized to fit an 8 GB GPU. The periodic transient prolongs its whole trajectory back to the full
    # nodal space (SemidiscreteTimeBlock.solve) and reverse-mode holds every prolonged step, so the PEB
    # mesh sets the peak. Shrink the SEED first -- printed(grid=) defaults to a 128x128 image, 16x finer
    # than this gradient check needs -- then take a mesh to match: at grid=32 the pixel pitch is 0.0875,
    # so 0.3 stays inside the 4-pixel coarseness guard while being ~8x fewer tets than the 0.15 default.
    # Coarsening the mesh alone would instead trip that guard, i.e. test the regime the library warns about.
    resist = jno.litho.CAResist(
        n=8, t_peb=10.0, steps=3, dill_c=4.0, diffusion_length=(0.12, 0.08), film=film, mesh_size=0.3
    )
    node = (
        jno.rcwa(cons, orders=30, grid=24, params={"ep": 2.0}).solve().printed(NA=0.6, source=0.5, grid=32, resist=resist)
    )
    assert isinstance(node, FunctionCall)
    g, fd = _grad_vs_fd(node, ep)
    assert g == pytest.approx(fd, rel=5e-3)
    assert abs(g) > 1e-4
