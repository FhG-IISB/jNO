"""``jno.fv(...)``: the finite-volume scheme of a flux, passed as ``scheme=`` to its outer derivative.

Based on:
  Toro, E. F. (2009). Riemann Solvers and Numerical Methods for Fluid Dynamics (3rd ed.). Springer --
  the Rusanov (local Lax-Friedrichs) and HLL fluxes (ch. 10), and the Roe flux (ch. 11).
  Rusanov, V. V. (1961). The calculation of the interaction of non-stationary shock waves with barriers.
  Zh. Vychisl. Mat. Mat. Fiz. 1, 267-279.
  Harten, A., Lax, P. D., & van Leer, B. (1983). On upstream differencing and Godunov-type schemes for
  hyperbolic conservation laws. SIAM Review 25(1), 35-61 -- HLL, and the entropy fix.
  Roe, P. L. (1981). Approximate Riemann solvers, parameter vectors, and difference schemes. Journal of
  Computational Physics 43(2), 357-372.
  Darwish, M. S., & Moukalled, F. (2003). TVD schemes for unstructured grids. International Journal of
  Heat and Mass Transfer 46(4), 599-611 -- the face ratio that carries any ψ(r) limiter to an
  unstructured mesh.
  van Albada, G. D., van Leer, B., & Roberts, W. W. (1982). A comparative study of computational methods
  in cosmic gas dynamics. Astronomy and Astrophysics 108, 76-84.
  Barth, T. J., & Jespersen, D. C. (1989). The design and application of upwind schemes on unstructured
  meshes. AIAA Paper 89-0366.
  Venkatakrishnan, V. (1995). Convergence to steady state solutions of the Euler equations on
  unstructured grids with limiters. Journal of Computational Physics 118(1), 120-130.

Nothing here knows a particular PDE. The fluxes need only the gathered normal flux ``F·n`` of the system
and its Jacobian, which ``jax`` gives; the generic Roe flux takes ``|A|`` as the matrix square root of
``A² + δ²I`` (no eigendecomposition, so it runs on any device and differentiates), which is Harten's
entropy fix built in.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["FVScheme", "fv"]

_FLUXES = ("rusanov", "hll", "roe")
_LIMITERS = ("vanalbada", "minmod", "vanleer", "superbee", "barth", "venkat")
_RECON = ("lsq", "green_gauss", "none")
_AVERAGE = ("arithmetic", "harmonic", "exact")


class FVScheme(str):
    """A finite-volume flux scheme (see :func:`fv`). A ``str`` so it rides on ``scheme=`` like a
    ``jno.fd(...)`` stencil."""

    def __new__(
        cls,
        *,
        flux="rusanov",
        limiter="vanalbada",
        recon="lsq",
        average=None,
        speed=None,
        equilibrium=None,
        entropy_fix=0.1,
        venkat_k=0.3,
    ):
        def name(v):
            return v if isinstance(v, str) or v is None else f"#{id(v)}"

        parts = [f"flux={name(flux)}", f"limiter={name(limiter)}", f"recon={recon}"]
        if average is not None:
            parts.append(f"average={average}")
        if speed is not None:
            parts.append(f"speed=#{id(speed)}")
        if equilibrium is not None:
            parts.append(f"equilibrium=#{id(equilibrium)}")
        self = super().__new__(cls, "finite_difference:fv(" + ",".join(parts) + ")")
        if isinstance(flux, str) and flux not in _FLUXES:
            raise ValueError(f"jno.fv: flux={flux!r} is not one of {_FLUXES} (or a callable flux(qL, qR, n, Fn)).")
        if isinstance(limiter, str) and limiter not in _LIMITERS:
            raise ValueError(f"jno.fv: limiter={limiter!r} is not one of {_LIMITERS}, None, or a callable ψ(r).")
        if recon not in _RECON:
            raise ValueError(f"jno.fv: recon={recon!r} is not one of {_RECON}.")
        if average is not None and average not in _AVERAGE:
            raise ValueError(f"jno.fv: average={average!r} is not one of {_AVERAGE}.")
        self.flux, self.limiter, self.recon, self.average = flux, limiter, recon, average
        self.speed, self.equilibrium, self.entropy_fix = speed, equilibrium, float(entropy_fix)
        self.venkat_k = float(venkat_k)
        return self

    def __reduce__(self):
        return (
            _rebuild,
            (
                self.flux,
                self.limiter,
                self.recon,
                self.average,
                self.speed,
                self.equilibrium,
                self.entropy_fix,
                self.venkat_k,
            ),
        )

    # -- limiters -----------------------------------------------------------------------------------
    def increment(self, a, b, eps2):
        """``a·ψ(b/a)``: the limited increment of a face state over its upwind cell value, written without the
        division so that it stays smooth where ``a = q_D − q_C`` vanishes. ``b = 2∇q_C·d − a`` is the upwind
        difference (Darwish & Moukalled 2003), ``eps2`` the smoothing of van Albada's form."""
        lim = self.limiter
        if lim is None:
            return b  # ψ(r) = r: the unlimited second-order state
        if callable(lim):
            return a * lim(b * a / (a * a + eps2))
        if lim == "vanalbada":
            return (a * (b * b + eps2) + b * (a * a + eps2)) / (a * a + b * b + 2.0 * eps2)
        same = a * b > 0
        if lim == "minmod":
            return jnp.where(same, jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.abs(b)), 0.0)
        if lim == "vanleer":
            return jnp.where(same, 2.0 * a * b / jnp.where(same, a + b, 1.0), 0.0)
        if lim == "superbee":
            big = jnp.maximum(jnp.minimum(2.0 * jnp.abs(a), jnp.abs(b)), jnp.minimum(jnp.abs(a), 2.0 * jnp.abs(b)))
            return jnp.where(same, jnp.sign(a) * big, 0.0)
        raise AssertionError(lim)

    @property
    def multidimensional(self):
        """Whether the limiter bounds the reconstruction by the neighbours' extrema (Barth-Jespersen,
        Venkatakrishnan) rather than by a ψ(r) of each face."""
        return self.limiter in ("barth", "venkat")


def _rebuild(flux, limiter, recon, average, speed, equilibrium, entropy_fix, venkat_k):
    return FVScheme(
        flux=flux,
        limiter=limiter,
        recon=recon,
        average=average,
        speed=speed,
        equilibrium=equilibrium,
        entropy_fix=entropy_fix,
        venkat_k=venkat_k,
    )


def fv(
    *,
    flux="rusanov",
    limiter="vanalbada",
    recon="lsq",
    average=None,
    speed=None,
    equilibrium=None,
    entropy_fix=0.1,
    venkat_k=0.3,
):
    """The finite-volume scheme of a flux, for ``scheme=`` on the flux's outer derivative::

        (bx*ui).d(x, scheme=jno.fv(flux="hll", limiter="minmod")) + (by*ui).d(y, scheme=jno.fv(flux="hll", limiter="minmod"))

    One equation's flux terms form one normal flux ``F·n``, so every axis of an equation (and every
    equation of a system) must carry the same scheme.

    Args:
        flux: the numerical flux. ``"rusanov"`` (default: central flux plus the spectral radius of
            ``∂(F·n)/∂q`` times the jump, exact upwinding for a linear flux), ``"hll"`` (the two-wave
            HLL flux, wave speeds from the extreme eigenvalues), ``"roe"`` (``½(F_L + F_R) − ½|A|(q_R −
            q_L)`` with ``A = ∂(F·n)/∂q`` at the mean state, for any system), or a callable
            ``flux(qL, qR, n, Fn)`` returning the normal flux per face, where ``Fn(q)`` is the gathered
            normal flux.
        limiter: how the second-order reconstruction is limited. ``"vanalbada"`` (default, smooth, so
            Newton and adjoints see a differentiable residual), ``"minmod"``, ``"vanleer"``,
            ``"superbee"`` -- ψ(r) limiters carried to unstructured faces (Darwish & Moukalled 2003) --
            ``"barth"`` and ``"venkat"`` (multidimensional, bounded by the neighbours' values),
            ``None`` (unlimited), or a callable ``ψ(r)``.
        recon: the reconstruction of the face states: ``"lsq"`` (least-squares gradients, default),
            ``"green_gauss"``, ``"none"`` (first order).
        average: how a coefficient in a diffusive flux is taken at a face: evaluated there from the
            interpolated state (default), ``"arithmetic"`` or ``"harmonic"`` mean of the two cell values,
            or ``"exact"`` (evaluated at the face; the same as the default for a formula).
        speed: the wave speed as an expression of the state, in place of the spectral radius.
        equilibrium: a steady state of the system to preserve exactly (well-balancing): the discrete
            residual of this state is subtracted (Berberich, Chandrashekar & Klingenberg 2021).
        entropy_fix: Harten's δ for ``"roe"``, as a fraction of the spectral radius.
        venkat_k: Venkatakrishnan's ``K``, with ``ε² = (K h)³``: how far a smooth extremum may pass the
            neighbours' bounds (0.3 kept a unit step within 0.1% on a triangulation; his 5 let it overshoot
            by 13%).
    """
    return FVScheme(
        flux=flux,
        limiter=limiter,
        recon=recon,
        average=average,
        speed=speed,
        equilibrium=equilibrium,
        entropy_fix=entropy_fix,
        venkat_k=venkat_k,
    )
