"""Spectral methods: Fourier / Chebyshev / Legendre unknowns (``dom.unknown(space="spectral", order=...)``)."""

from .bases import Chebyshev, Fourier, Legendre
from .grid import Grid
from .space import SpectralSpace

__all__ = ["Chebyshev", "Fourier", "Grid", "Legendre", "SpectralSpace"]
