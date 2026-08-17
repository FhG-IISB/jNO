"""Resist models for ``jno.rcwa`` computational lithography.

A **resist** turns the optical exposure at the wafer (``sol.expose(...)``) into a developed pattern. It is
any callable ``exposure -> developed field`` -- so it is applied with ``exposure.develop(resist)`` and new
models plug in without touching the imaging code. This module ships the fast, differentiable design-loop
model :class:`Threshold` and the rigorous 3-species reaction-diffusion PEB model :class:`CAResist`, which
plug into the same ``develop`` seam. ``CAResist`` develops the 2-D aerial image by default, or -- given a
:class:`Film` -- the depth-resolved standing-wave bulk image (:meth:`_Exposure.bulk`) as a full 3-D
``(x, y, z)`` reaction-diffusion bake (the species diffuse through the film thickness too).
"""

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Film:
    """Resist-film stack for the standing-wave **bulk image** (:meth:`_Exposure.bulk`): a resist of index
    ``n_resist`` (complex ⇒ absorption) and ``thickness`` on a substrate ``n_substrate``, under a top medium
    ``n_top``, sampled at ``nz`` depths. Same length unit as the mask period."""

    n_resist: complex
    thickness: float
    n_substrate: complex = 1.0
    n_top: float = 1.0
    nz: int = 16


class Threshold:
    """Constant-threshold resist with a linear (Gaussian) post-exposure-bake (PEB) diffusion -- the fast,
    differentiable resist that drives OPC / ILT / SMO.

    Development blurs the aerial image by the PEB acid-diffusion length (a periodic Gaussian, the bake heat
    kernel), then applies a soft constant threshold ``sigmoid(steepness · (I_bake − threshold))`` ∈ ``[0, 1]``
    (1 = clears, positive tone) -- a differentiable stand-in for the printed contour. Linear-diffusion +
    constant-threshold model: Poonawala & Milanfar, *IEEE Trans. Image Process.* **16**, 774 (2007); PEB
    diffusion after Mack, *Fundamental Principles of Optical Lithography* (2007).

    Parameters
    ----------
    threshold:
        Dose-to-clear fraction on the aerial-intensity scale (an open frame images to ≈ 1). Raising it
        shrinks a bright feature -- the knob that sets printed CD.
    diffusion:
        PEB acid-diffusion length (same length unit as the geometry; ``0`` = no bake).
    steepness:
        Development contrast -- the sigmoid sharpness (larger → a harder threshold / steeper resist).
    """

    def __init__(self, threshold=0.3, diffusion=0.0, steepness=50.0):
        self.threshold = float(threshold)
        self.diffusion = float(diffusion)
        self.steepness = float(steepness)

    def __call__(self, exposure):
        """Develop an exposure into a ``[0, 1]`` resist image. Needs ``exposure.intensity()`` (the aerial
        image) and ``exposure.period`` (for the diffusion length scale)."""
        return _develop(exposure.intensity(), self.threshold, self.diffusion, self.steepness, exposure.period)


def _develop(img, threshold, diffusion, steepness, period):
    """Constant-threshold resist with a linear (Gaussian) PEB diffusion: blur the aerial image by the acid-
    diffusion length (periodic, in Fourier space -- the image is one period), then soft-threshold. Axis 0 is
    x (period ``period[0]``), axis 1 is y. Differentiable; ``diffusion == 0`` skips the blur exactly."""
    if diffusion > 0:  # linear PEB diffusion == a periodic Gaussian blur (heat kernel over the bake)
        fx = jnp.fft.fftfreq(img.shape[0], d=period[0] / img.shape[0])
        fy = jnp.fft.fftfreq(img.shape[1], d=period[1] / img.shape[1])
        FX, FY = jnp.meshgrid(fx, fy, indexing="ij")
        ker = jnp.exp(-2.0 * (jnp.pi * diffusion) ** 2 * (FX**2 + FY**2))
        img = jnp.real(jnp.fft.ifft2(jnp.fft.fft2(img) * ker))
    return jax.nn.sigmoid(steepness * (img - threshold))


class CAResist:
    """Rigorous **chemically-amplified-resist** post-exposure bake -- the 3-species reaction-diffusion PEB
    (dos Santos, *Simulation of post-exposure bake reactions using PINNs for chemically amplified resists*),
    coupled to an RCWA exposure. The exposure's aerial image sets the **latent acid** (Dill exposure kinetics),
    then inhibitor ``M``, acid ``A`` and quencher ``B`` react-and-diffuse over the bake, and the developed
    ``1 − M`` (deprotection) is the printed pattern::

        M_t = −k1 M A − k2 M                 (inhibitor    — immobile)
        A_t =  D_A ΔA − k3 A − k4 A B        (acid         — diffuses)
        B_t =  D_B ΔB − k4 A B − k5 B        (base quencher — diffuses)

    Authored as one transient :func:`jno.fem` system (the sole FEM entry) on a doubly-periodic film mesh, so
    it is a **drop-in resist** for ``exposure.develop(...)`` -- the rigorous counterpart of :class:`Threshold`.
    It is heavier (a nonlinear multifield transient solve), for verification rather than the fast design loop.

    Parameters
    ----------
    n, t_peb, steps:
        Film-mesh resolution per side, bake time, and number of (backward-Euler) time steps.
    source_chunk:
        Accumulate the bulk image's Abbe sum this many source points at a time (``None`` = all at once).
        Bounds peak memory when a ``film`` is given; see :meth:`jno.rcwa._Exposure.bulk`.
    mesh_size:
        PEB mesh element size (same length unit as the period). ``None`` = ``min(Px, Py, thickness)/4``,
        which resolves the film thickness but *not* in-plane features of a wider clip -- set it to a
        fraction of your smallest feature when the returned volume looks blocky.
    k:
        Rate constants ``(k1, k2, k3, k4, k5)``. ``k4`` is the bilinear acid-base neutralization (the stiff,
        genuinely-nonlinear coupling).
    diffusion_length:
        Acid/base diffusion lengths ``(ρ_A, ρ_B)`` (same length unit as the mask period); ``D = ρ²/(2 t_peb)``.
    dill_c, dose:
        Dill exposure-rate constant and exposure dose -- the latent acid is
        ``A(t=0) = 1 − exp(−dill_c · dose · I)`` from the aerial intensity ``I``. **Calibrate ``dill_c·dose``
        to the exposure scale** so the acid *straddles* ``quencher`` across the pattern (roughly
        ``dill_c·dose·median(I) ≈ −ln(1 − quencher)``): a seed that saturates (``A0 ≈ 1`` everywhere) or
        starves (``A0 ≈ 0``) bakes to a **flat** developed field with no printed contrast.
    quencher:
        Uniform initial base-quencher loading ``B(t=0)``. The developed pattern forms at the contour where
        the latent acid crosses this level (excess acid deprotects and clears; below it the resist stays).
    tone:
        ``"positive"`` (default) returns the developed/soluble fraction ``1 − M``; ``"negative"`` returns ``M``.
    film:
        ``None`` (default) → **2-D** PEB driven by the aerial image (matching the dos Santos 2-D model),
        returning a ``(n, n)`` pattern. A :class:`Film` → **3-D** ``(x, y, z)`` PEB on a ``jno.Shape`` box:
        the acid is seeded from the depth-resolved standing-wave bulk image (:meth:`_Exposure.bulk`), the
        species diffuse in x, y *and* z, and a developed ``(n, n, film.nz)`` volume is returned.
    """

    def __init__(
        self,
        *,
        n=48,
        t_peb=45.0,
        steps=30,
        k=(0.5, 0.005, 0.005, 5.0, 0.005),
        diffusion_length=(12.0, 8.0),
        dill_c=1.0,
        dose=1.0,
        quencher=0.4,
        tone="positive",
        film=None,
        source_chunk=None,
        mesh_size=None,
    ):
        if tone not in ("positive", "negative"):
            raise ValueError(f"tone must be 'positive' or 'negative', got {tone!r}")
        if len(k) != 5:
            raise ValueError(f"k must be (k1, k2, k3, k4, k5), got {len(k)} values")
        self.n, self.t_peb, self.steps = int(n), float(t_peb), int(steps)
        self.k = tuple(float(v) for v in k)
        self.rho_a, self.rho_b = (float(v) for v in diffusion_length)
        self.dill_c, self.dose, self.quencher, self.tone = float(dill_c), float(dose), float(quencher), tone
        self.film = film
        # Bounds the peak memory of the bulk image this resist reads: the Abbe sum over source points is
        # accumulated `source_chunk` points at a time rather than materialising the whole
        # (n_source, nz, grid, grid) stack. The 3-D PEB is exactly the caller that needs it -- it reads a
        # depth-resolved image over the full film -- so leaving it unthreaded made the bulk chunking apply
        # only to direct `bulk()` calls and not to the resist model that mainly uses them.
        self.source_chunk = source_chunk
        # PEB mesh element size. The default `min(Px, Py, thickness)/4` gives only ~4 elements through
        # the thinnest dimension, which for a mask clip much wider than the film is far too coarse to
        # resolve in-plane features -- the returned volume is then dominated by the scatter-mean regrid
        # (empty output cells fall back to the global mean, which reads as blocky noise). Set this to
        # resolve the smallest feature you care about.
        self.mesh_size = None if mesh_size is None else float(mesh_size)

    def __call__(self, exposure):
        """Develop an exposure through the reaction-diffusion PEB. With no ``film`` this reads the 2-D aerial
        image (``exposure.intensity()``) → ``(n, n)``; with a :class:`Film` it reads the depth-resolved bulk
        image (``exposure.bulk(film)``) and solves the 3-D PEB → ``(n, n, film.nz)``. Both in ``[0, 1]``."""
        if self.film is None:
            return _peb_develop(exposure.intensity(), exposure.period, self)
        return _peb_develop_3d(exposure.bulk(self.film, source_chunk=self.source_chunk), exposure.period, self.film, self)


def _sample_periodic(img, x, y, period):
    """Differentiable periodic bilinear sample of a ``(G, G)`` image (axis 0 = x, axis 1 = y, over one
    ``period``) at node coordinates ``(x, y)``."""
    gx, gy = (x / period[0] % 1.0) * img.shape[0], (y / period[1] % 1.0) * img.shape[1]
    x0, y0 = jnp.floor(gx).astype(int) % img.shape[0], jnp.floor(gy).astype(int) % img.shape[1]
    x1, y1 = (x0 + 1) % img.shape[0], (y0 + 1) % img.shape[1]
    fx, fy = gx - jnp.floor(gx), gy - jnp.floor(gy)
    return (
        img[x0, y0] * (1 - fx) * (1 - fy)
        + img[x1, y0] * fx * (1 - fy)
        + img[x0, y1] * (1 - fx) * fy
        + img[x1, y1] * fx * fy
    )


def _peb_develop(img, period, r):
    """Build and solve the 3-species reaction-diffusion PEB (weak form after dos Santos) on a doubly-periodic
    film mesh, seeded by the Dill latent acid from ``img``, and return the developed ``(n, n)`` pattern.
    ``jno`` is imported lazily so this resist module has no import cycle with the package."""
    import jno
    from jno.trace_evaluator import TraceEvaluator

    Px, Py = period
    n, (k1, k2, k3, k4, k5) = r.n, r.k
    d_a, d_b = r.rho_a**2 / (2.0 * r.t_peb), r.rho_b**2 / (2.0 * r.t_peb)

    dom = (
        jno.Shape.rect(0.0, 0.0, Px, Py)
        .structured(n=n)
        .domain(
            time=(0.0, r.t_peb, r.steps),
            compute_mesh_connectivity=False,
        )
    )
    ex, ey = 1e-6 * Px, 1e-6 * Py
    for nm, pred in {
        "left": lambda x, y: x < ex,
        "right": lambda x, y: x > Px - ex,
        "bottom": lambda x, y: y < ey,
        "top": lambda x, y: y > Py - ey,
    }.items():
        dom.tag(nm, pred)
    M, pM = dom.fem_symbols(names=("M", "pM"))
    A, pA = dom.fem_symbols(names=("A", "pA"))
    B, pB = dom.fem_symbols(names=("B", "pB"))
    xi, yi, ti = dom.variable("interior", split=True)
    ci = dom.variable("initial", split=True)
    xl, yl, _ = dom.variable("left", split=True)
    xr, yr, _ = dom.variable("right", split=True)
    xb, yb, _ = dom.variable("bottom", split=True)
    xt, yt, _ = dom.variable("top", split=True)
    Mi, qM = M.bind(x=xi, y=yi, t=ti), pM.bind(x=xi, y=yi, t=ti)
    Ai, qA = A.bind(x=xi, y=yi, t=ti), pA.bind(x=xi, y=yi, t=ti)
    Bi, qB = B.bind(x=xi, y=yi, t=ti), pB.bind(x=xi, y=yi, t=ti)

    def acid0(x, y):  # Dill latent acid from the aerial intensity, sampled at the film nodes
        return 1.0 - jnp.exp(-r.dill_c * r.dose * _sample_periodic(img, x, y, period))

    A0 = jno.fn(acid0, [ci[0], ci[1]])
    fem = jno.fem(
        [
            Mi.t * qM + k1 * Mi * Ai * qM + k2 * Mi * qM,  # inhibitor: immobile, reaction only
            Ai.t * qA + d_a * (Ai.x * qA.x + Ai.y * qA.y) + k3 * Ai * qA + k4 * Ai * Bi * qA,  # acid: diffuse + react
            Bi.t * qB + d_b * (Bi.x * qB.x + Bi.y * qB.y) + k4 * Ai * Bi * qB + k5 * Bi * qB,  # quencher: diffuse + react
            M(xl, yl) - M(xr, yr),
            M(xb, yb) - M(xt, yt),
            A(xl, yl) - A(xr, yr),
            A(xb, yb) - A(xt, yt),
            B(xl, yl) - B(xr, yr),
            B(xb, yb) - B(xt, yt),
            M(ci[0], ci[1]) - 1.0,
            A(ci[0], ci[1]) - A0,
            B(ci[0], ci[1]) - r.quencher,
        ]
    )
    node = fem.solve()
    traj = TraceEvaluator({}).evaluate(node.expr if hasattr(node, "expr") else node, context={})
    off = fem.offsets
    m_final = traj[-1][off[0] : off[1]]  # inhibitor M on the (periodic-reduced) film nodes
    coords = np.asarray(fem.field_points[0])[:, :2]
    hx, hy = Px / n, Py / n
    ix = np.round(coords[:, 0] / hx).astype(int) % n
    iy = np.round(coords[:, 1] / hy).astype(int) % n
    grid = jnp.zeros((n, n)).at[ix, iy].set(m_final)  # regrid M onto one period (differentiable in M)
    developed = (1.0 - grid) if r.tone == "positive" else grid
    return jnp.clip(developed, 0.0, 1.0)  # a developed fraction is physical in [0, 1]


def _sample_bulk(vol, x, y, z, period, thickness):
    """Differentiable trilinear sample of a ``(G, G, nz)`` bulk image -- periodic in x, y (over ``period``),
    clamped in z (over ``thickness``) -- at node coordinates ``(x, y, z)``."""
    G, nz = vol.shape[0], vol.shape[2]
    gx, gy = (x / period[0] % 1.0) * G, (y / period[1] % 1.0) * G
    gz = jnp.clip(z / thickness, 0.0, 1.0) * (nz - 1)
    x0, y0 = jnp.floor(gx).astype(int) % G, jnp.floor(gy).astype(int) % G
    x1, y1 = (x0 + 1) % G, (y0 + 1) % G
    z0 = jnp.clip(jnp.floor(gz).astype(int), 0, nz - 1)
    z1 = jnp.clip(z0 + 1, 0, nz - 1)
    fx, fy, fz = gx - jnp.floor(gx), gy - jnp.floor(gy), gz - jnp.floor(gz)

    def face(zi):
        return (
            vol[x0, y0, zi] * (1 - fx) * (1 - fy)
            + vol[x1, y0, zi] * fx * (1 - fy)
            + vol[x0, y1, zi] * (1 - fx) * fy
            + vol[x1, y1, zi] * fx * fy
        )

    return face(z0) * (1 - fz) + face(z1) * fz


def _regrid3d(vals, coords, n, nz, period, thickness):
    """Interpolate unstructured node values onto a regular ``(n, n, nz)`` grid by **barycentric**
    interpolation (differentiable in ``vals``): each output point takes a weighted combination of its
    containing tetrahedron's four nodes.

    This replaces a nearest-cell scatter-mean, which was only valid when ``n*n*nz << n_nodes``: output
    cells containing no node fell back to the GLOBAL MEAN, so at the default ``n=48, nz=24`` (55k cells)
    against a ~1.8k-node PEB mesh, ~97% of the returned volume was a single constant. That read as blocky
    noise and got WORSE as ``n`` was raised, since more cells meant more of them empty. Barycentric
    interpolation is independent of the node count, so ``n`` now sets the output resolution and nothing
    else. Points outside the hull fall back to the nearest node (see ``jno.rcwa._bary_weights``).
    """
    from jno.rcwa import _bary_weights  # lazy: jno.rcwa imports jno.litho at module scope

    Px, Py = period
    xs = (np.arange(n) + 0.5) / n * Px
    ys = (np.arange(n) + 0.5) / n * Py
    zs = (np.arange(nz) + 0.5) / nz * float(thickness)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    ids, w = _bary_weights(np.asarray(coords, dtype=float), pts)
    g = jnp.sum(jnp.asarray(w) * jnp.asarray(vals)[jnp.asarray(ids)], axis=1)
    return g.reshape(n, n, nz)


def _peb_develop_3d(vol, period, film, r):
    """3-D reaction-diffusion PEB on a ``jno.Shape`` box (periodic in x, y; free in z), seeded by the Dill
    latent acid from the standing-wave bulk image ``vol`` (``(G, G, nz)``). The species diffuse in x, y and z;
    returns the developed ``(n, n, film.nz)`` volume. ``jno`` is imported lazily (no package import cycle)."""
    import jno
    from jno.trace_evaluator import TraceEvaluator

    Px, Py = period
    d, n = float(film.thickness), r.n
    k1, k2, k3, k4, k5 = r.k
    if not jax.config.jax_enable_x64:  # measured: every seed tried returns an all-NaN volume in float32
        warnings.warn(
            "jno.litho.CAResist: the 3-D PEB is being solved in float32. The bilinear acid-quencher "
            "term (k4) makes the backward-Euler system stiff enough that float32 diverges to NaN across "
            "the whole volume -- silently, since the NaNs are only visible in the returned array. Enable "
            "jax.config.update('jax_enable_x64', True) before calling.",
            stacklevel=2,
        )
    d_a, d_b = r.rho_a**2 / (2.0 * r.t_peb), r.rho_b**2 / (2.0 * r.t_peb)
    # Mesh size. The seed image `vol` is (G, G, nz) over the period, so its pixel pitch Px/G is the
    # finest structure the PEB can possibly be given -- a mesh coarser than that discards exposure data
    # before solving. Sizing by the film thickness alone (the old `min(Px, Py, d)/4`) knows about the
    # THINNEST dimension and nothing about the pattern, which for a clip much wider than the film is the
    # wrong constraint: at Px = 1.4 um, G = 96 and d = 0.3 um it gave 75 nm elements for a 14.6 nm input.
    # Default to the tighter of the two, capped at 2 input pixels so the mesh does not explode.
    gx = int(jnp.shape(vol)[0])  # shape only -- `vol` is a tracer under jit/grad (ILT), never realise it
    px_in = Px / max(gx, 1)
    size = float(r.mesh_size) if getattr(r, "mesh_size", None) else min(min(Px, Py, d) / 4.0, 2.0 * px_in)
    if size > 4.0 * px_in:  # never silently throw away most of the seed image
        warnings.warn(
            f"jno.litho.CAResist: PEB mesh {size:.4g} is {size / px_in:.1f}x coarser than the exposure "
            f"image it is seeded from ({px_in:.4g} per pixel), so most of that image is averaged away "
            f"before the bake. Pass a smaller mesh_size, or a coarser exposure grid, if that is not "
            f"intended.",
            stacklevel=2,
        )

    dom = jno.Shape.box(0.0, 0.0, 0.0, Px, Py, d, size=size).domain(
        time=(0.0, r.t_peb, r.steps), compute_mesh_connectivity=False
    )
    ex, ey = 1e-6 * Px, 1e-6 * Py
    for nm, pred in {
        "left": lambda x, y, z: x < ex,
        "right": lambda x, y, z: x > Px - ex,
        "front": lambda x, y, z: y < ey,
        "back": lambda x, y, z: y > Py - ey,
    }.items():
        dom.tag(nm, pred)
    dom._remesh_periodic([("left", "right"), ("front", "back")])  # conforming x,y faces for the nodal ties
    M, pM = dom.fem_symbols(names=("M", "pM"))
    A, pA = dom.fem_symbols(names=("A", "pA"))
    B, pB = dom.fem_symbols(names=("B", "pB"))
    xi, yi, zi, ti = dom.variable("interior", split=True)
    ci = dom.variable("initial", split=True)
    xl, yl, zl, _ = dom.variable("left", split=True)
    xr, yr, zr, _ = dom.variable("right", split=True)
    xf, yf, zf, _ = dom.variable("front", split=True)
    xk, yk, zk, _ = dom.variable("back", split=True)
    Mi, qM = M.bind(x=xi, y=yi, z=zi, t=ti), pM.bind(x=xi, y=yi, z=zi, t=ti)
    Ai, qA = A.bind(x=xi, y=yi, z=zi, t=ti), pA.bind(x=xi, y=yi, z=zi, t=ti)
    Bi, qB = B.bind(x=xi, y=yi, z=zi, t=ti), pB.bind(x=xi, y=yi, z=zi, t=ti)

    def acid0(x, y, z):  # Dill latent acid from the bulk image at the film nodes
        return 1.0 - jnp.exp(-r.dill_c * r.dose * _sample_bulk(vol, x, y, z, period, d))

    A0 = jno.fn(acid0, [ci[0], ci[1], ci[2]])
    fem = jno.fem(
        [
            Mi.t * qM + k1 * Mi * Ai * qM + k2 * Mi * qM,  # inhibitor: immobile, reaction only
            Ai.t * qA + d_a * (Ai.x * qA.x + Ai.y * qA.y + Ai.z * qA.z) + k3 * Ai * qA + k4 * Ai * Bi * qA,  # acid
            Bi.t * qB + d_b * (Bi.x * qB.x + Bi.y * qB.y + Bi.z * qB.z) + k4 * Ai * Bi * qB + k5 * Bi * qB,  # quencher
            M(xl, yl, zl) - M(xr, yr, zr),
            M(xf, yf, zf) - M(xk, yk, zk),
            A(xl, yl, zl) - A(xr, yr, zr),
            A(xf, yf, zf) - A(xk, yk, zk),
            B(xl, yl, zl) - B(xr, yr, zr),
            B(xf, yf, zf) - B(xk, yk, zk),
            M(ci[0], ci[1], ci[2]) - 1.0,
            A(ci[0], ci[1], ci[2]) - A0,
            B(ci[0], ci[1], ci[2]) - r.quencher,
        ]
    )
    node = fem.solve()
    traj = TraceEvaluator({}).evaluate(node.expr if hasattr(node, "expr") else node, context={})
    m_final = traj[-1][fem.offsets[0] : fem.offsets[1]]
    coords = np.asarray(fem.field_points[0])[:, :3]
    grid = _regrid3d(m_final, coords, n, int(film.nz), period, d)
    developed = (1.0 - grid) if r.tone == "positive" else grid
    # a developed fraction is physical in [0, 1]; the P1 reaction-diffusion can slightly overshoot at sharp
    # acid gradients (non-lumped mass), so clip to the documented range.
    return jnp.clip(developed, 0.0, 1.0)
