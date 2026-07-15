"""Resist models for ``jno.rcwa`` computational lithography.

A **resist** turns the optical exposure at the wafer (``sol.expose(...)``) into a developed pattern. It is
any callable ``exposure -> developed field`` -- so it is applied with ``exposure.develop(resist)`` and new
models plug in without touching the imaging code. This module ships the fast, differentiable design-loop
model :class:`Threshold` and the rigorous 3-species reaction-diffusion PEB model :class:`CAResist`, which
plug into the same ``develop`` seam. ``CAResist`` currently drives its latent acid from the aerial image;
the exposure's :meth:`_Exposure.bulk` gives the depth-resolved standing-wave field (a :class:`Film` stack),
so consuming it in a 3-D PEB solve is the next step.
"""

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
    k:
        Rate constants ``(k1, k2, k3, k4, k5)``. ``k4`` is the bilinear acid-base neutralization (the stiff,
        genuinely-nonlinear coupling).
    diffusion_length:
        Acid/base diffusion lengths ``(ρ_A, ρ_B)`` (same length unit as the mask period); ``D = ρ²/(2 t_peb)``.
    dill_c, dose:
        Dill exposure-rate constant and exposure dose -- the latent acid is
        ``A(t=0) = 1 − exp(−dill_c · dose · I)`` from the aerial intensity ``I``.
    quencher:
        Uniform initial base-quencher loading ``B(t=0)``.
    tone:
        ``"positive"`` (default) returns the developed/soluble fraction ``1 − M``; ``"negative"`` returns ``M``.

    First cut: the latent acid is driven by the 2-D aerial image (matching the dos Santos 2-D model). The
    depth-resolved standing-wave bulk image is available from :meth:`_Exposure.bulk`; consuming it in a 3-D
    (x, y, z) PEB solve is the next refinement.
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
    ):
        if tone not in ("positive", "negative"):
            raise ValueError(f"tone must be 'positive' or 'negative', got {tone!r}")
        if len(k) != 5:
            raise ValueError(f"k must be (k1, k2, k3, k4, k5), got {len(k)} values")
        self.n, self.t_peb, self.steps = int(n), float(t_peb), int(steps)
        self.k = tuple(float(v) for v in k)
        self.rho_a, self.rho_b = (float(v) for v in diffusion_length)
        self.dill_c, self.dose, self.quencher, self.tone = float(dill_c), float(dose), float(quencher), tone

    def __call__(self, exposure):
        """Develop an exposure through the reaction-diffusion PEB. Reads ``exposure.intensity()`` (the aerial
        image) and ``exposure.period``; returns a developed ``(n, n)`` pattern in ``[0, 1]``."""
        return _peb_develop(exposure.intensity(), exposure.period, self)


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

    dom = jno.domain(
        constructor=jno.domain.equi_distant_rect(x_range=(0.0, Px), y_range=(0.0, Py), nx=n, ny=n),
        time=(0.0, r.t_peb, r.steps),
        compute_mesh_connectivity=False,
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
    return 1.0 - grid if r.tone == "positive" else grid
