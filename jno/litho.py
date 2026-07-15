"""Resist models for ``jno.rcwa`` computational lithography.

A **resist** turns the optical exposure at the wafer (``sol.expose(...)``) into a developed pattern. It is
any callable ``exposure -> developed field`` -- so it is applied with ``exposure.develop(resist)`` and new
models plug in without touching the imaging code. This module ships the fast, differentiable design-loop
model :class:`Threshold`; a rigorous reaction-diffusion PEB model (``CAResist``) plugs into the same seam by
reading the exposure's angular spectrum instead of just its intensity.
"""

import jax
import jax.numpy as jnp


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
