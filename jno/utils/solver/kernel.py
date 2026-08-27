"""Integral-equation kernels — the shared core behind PEEC and any Green's-function method.

An integral equation couples every source element to every other through a Green's function
``G(r, r')``. Unlike a differential operator there is no sparsity to exploit: ``1/r`` has no cutoff,
and truncating far pairs deletes exactly the long-range coupling the method exists to capture. What
the matrix *does* have is structure, and which structure depends entirely on where the elements sit:

* **arbitrary positions** — no structure. The apply is a chunked O(N²) sum (:func:`pair_quadratic`).
* **a regular lattice** — ``G`` depends only on the index offset, so the matrix is block-Toeplitz
  Toeplitz-block and the apply is an FFT: O(N) memory, O(N log N) time (:func:`lattice_operator`).

Both are written in ``jnp``: differentiable, jittable, and GPU-capable. That is the whole reason this
lives here rather than in numpy — a dense pair sum is easy anywhere, but an integral-equation solve
whose *geometry* carries a gradient is what makes inverse design possible without a mesh.

The kernel itself is an ordinary function of distance, so a caller supplies physics rather than
plumbing::

    laplace = lambda r: 1.0 / r                     # 1/(4 pi r), constants folded by the caller
    helmholtz = lambda r: jnp.exp(1j * k * r) / r

**Self terms are the caller's responsibility and are not optional.** The diagonal
``G_aa = (1/V²) ∫∫ 1/|r-r'|`` is singular and shape-dependent; approximating a tetrahedron by an
equal-volume sphere overestimates it by 20 % (energy-weighted; up to 171 % on a sliver), because for
a fixed volume the sphere minimises the mean interior distance and so maximises ``<1/r>``. Passing
``self_g=None`` therefore raises rather than silently substituting a far-field value.

References
----------
Ruehli, *Equivalent Circuit Models for Three-Dimensional Multiconductor Systems*, IEEE Trans.
Microwave Theory Tech. 22(3):216, 1974 — the partial-element formulation.
Torchio, Peng, Moreno & Bettini, *A FFT-PEEC method*, IEEE Trans. Power Electron. 37(3), 2022 —
the translational-invariance argument behind :func:`lattice_operator`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp

__all__ = ["pair_quadratic", "lattice_kernel", "lattice_operator", "sphere_self"]


def sphere_self(volume):
    """Equal-volume-sphere self term ``6/(5R)`` — exact for a sphere, an UPPER BOUND otherwise.

    Provided because it is the standard first cut and is genuinely exact for spherical elements, but
    it is the wrong default for anything else: measured against direct integration over the cells of
    a real tetrahedral mesh it runs +20 % energy-weighted and +171 % on the worst slivers. Prefer a
    per-element integral when the elements are not spheres.
    """
    R = (3.0 * volume / (4.0 * jnp.pi)) ** (1.0 / 3.0)
    return 6.0 / (5.0 * R)


def pair_quadratic(pos, mom, g: Callable, self_g, chunk: int = 128):
    """``Σ_a Σ_b (mom_a · mom_b) g(|r_a − r_b|)`` over arbitrary element positions.

    Args:
        pos: ``(N, 3)`` element positions.
        mom: ``(N,)`` scalar or ``(N, d)`` vector density per element (for PEEC, ``J · volume``).
        g: the kernel, a callable of scalar distance.
        self_g: ``(N,)`` diagonal values ``g_aa``. Required — see the module docstring.
        chunk: rows per scan step. Bounds peak memory at ``chunk × N``; the reverse pass
            rematerialises rather than storing, so this also bounds the gradient's memory.

    The diagonal is excluded **by index**, not by a distance threshold. Distances come from
    ``r² = |a|² + |b|² − 2a·b`` to avoid forming an ``(chunk, N, 3)`` difference array, and that
    identity cancels catastrophically on the diagonal: for coordinates ~100 units from the origin it
    leaves ``r ≈ 3e-6`` rather than 0, which sails past any plausible epsilon and injects a huge
    spurious ``1/r``. Masking by index is exact and costs nothing.
    """
    if self_g is None:
        raise ValueError(
            "pair_quadratic: self_g is required. The diagonal of an integral operator is singular "
            "and shape-dependent; there is no safe default. Pass a per-element integral, or "
            "kernel.sphere_self(volume) if the elements really are spheres."
        )
    pos = jnp.asarray(pos)
    mom = jnp.asarray(mom)
    if mom.ndim == 1:
        mom = mom[:, None]
    n = pos.shape[0]
    npad = -(-n // chunk) * chunk

    P = jnp.zeros((npad, pos.shape[1]), pos.dtype).at[:n].set(pos)
    M = jnp.zeros((npad, mom.shape[1]), mom.dtype).at[:n].set(mom)
    n2 = (P * P).sum(1)
    rows = jnp.arange(chunk)
    cols = jnp.arange(npad)
    real = cols < n

    def body(acc, xs):
        start, pi, mi = xs
        idx = start + rows
        r2 = (pi * pi).sum(1)[:, None] + n2[None, :] - 2.0 * (pi @ P.T)
        # Excluded entries: the diagonal, and anything touching a pad row or column. Identified by
        # INDEX -- the distance identity cancels to ~3e-6 on the diagonal at large coordinates, so
        # no epsilon on `r` can find it reliably.
        excl = (idx[:, None] == cols[None, :]) | (~real)[None, :] | (~real[idx])[:, None]
        # Substitute BEFORE the sqrt. Masking afterwards would still differentiate the excluded
        # branch, and d(sqrt)/dx is infinite at 0, so `where` returns 0 * inf = NaN in reverse mode.
        r = jnp.sqrt(jnp.where(excl, 1.0, r2))
        # Zero the CONTRIBUTION rather than sending r to infinity: g(inf) -> 0 happens to hold for
        # 1/r, but a caller-supplied kernel carries no such promise.
        return acc + jnp.sum((mi @ M.T) * jnp.where(excl, 0.0, g(r))), None

    starts = jnp.arange(0, npad, chunk)
    acc, _ = jax.lax.scan(
        jax.checkpoint(body),
        jnp.zeros((), jnp.result_type(mom.dtype, pos.dtype)),
        (starts, P.reshape(-1, chunk, pos.shape[1]), M.reshape(-1, chunk, mom.shape[1])),
    )
    return acc + jnp.sum((mom * mom).sum(1) * jnp.asarray(self_g))


def lattice_kernel(n: Sequence[int], h: Sequence[float], g: Callable, self_g):
    """The BTTB generator on the doubled grid, ready for :func:`jax.numpy.fft.rfftn`.

    On a regular lattice ``G(i, j) = g(i − j)``, so the operator is block-Toeplitz Toeplitz-block and
    the number of independent coefficients falls from O(N²) to O(N). Embedding in a circulant of
    twice the size per axis turns the apply into a pointwise product in Fourier space.

    Index ``p`` along an axis of length ``2n`` maps to offset ``p`` below ``n`` and ``p − 2n`` above;
    the slot at ``p == n`` is unreachable by any real offset and is zeroed. Getting that wrap wrong
    on any single axis gives an answer that is wrong but entirely plausible-looking, which is why the
    test exercises an anisotropic cell rather than a cube.
    """
    axes = []
    for ni in n:
        p = jnp.arange(2 * ni)
        axes.append(jnp.where(p < ni, p, p - 2 * ni))
    off = jnp.meshgrid(*axes, indexing="ij")
    valid = jnp.ones(off[0].shape, bool)
    for o, ni in zip(off, n):
        valid &= jnp.abs(o) < ni
    d = jnp.sqrt(sum((o * hi) ** 2 for o, hi in zip(off, h)))
    body = jnp.where(d > 0, g(jnp.where(d > 0, d, 1.0)), self_g)
    return jnp.where(valid, body, 0.0)


def lattice_operator(n: Sequence[int], h: Sequence[float], g: Callable, self_g):
    """Return ``apply(x)`` performing the BTTB matvec by FFT.

    ``x`` has the lattice shape ``n``. The generator's transform is computed once here and closed
    over, so a Krylov solve pays for it a single time rather than per iteration.
    """
    n = tuple(int(v) for v in n)
    dbl = [2 * v for v in n]
    ghat = jnp.fft.rfftn(lattice_kernel(n, h, g, self_g))

    def apply(x):
        xp = jnp.zeros(dbl, x.dtype).at[tuple(slice(0, v) for v in n)].set(x)
        y = jnp.fft.irfftn(ghat * jnp.fft.rfftn(xp), s=dbl)
        return y[tuple(slice(0, v) for v in n)]

    return apply
