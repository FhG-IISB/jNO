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

__all__ = [
    "pair_quadratic",
    "pair_matrix",
    "lattice_kernel",
    "lattice_operator",
    "sphere_self",
    "bar_self",
    "wire_self",
]


def sphere_self(volume):
    """Equal-volume-sphere self term ``6/(5R)`` — exact for a sphere, an UPPER BOUND otherwise.

    Provided because it is the standard first cut and is genuinely exact for spherical elements, but
    it is the wrong default for anything else: measured against direct integration over the cells of
    a real tetrahedral mesh it runs +20 % energy-weighted and +171 % on the worst slivers. Prefer a
    per-element integral when the elements are not spheres.
    """
    R = (3.0 * volume / (4.0 * jnp.pi)) ** (1.0 / 3.0)
    return 6.0 / (5.0 * R)


def bar_self(length, width, thickness):
    """Self term of a straight rectangular bar (Ruehli, *IBM J. Res. Dev.* 16:470, 1972, eq. 12).

    ``Lp = (mu0 l/2pi)[ln(2l/(w+t)) + 0.5 + 0.2235 (w+t)/l]``, converted to the ``g_aa`` this module
    wants by ``Lp = (mu0/4pi) l^2 g_aa``. Valid for ``l >> w, t``.
    """
    s = width + thickness
    return 2.0 * (jnp.log(2.0 * length / s) + 0.5 + 0.2235 * s / length) / length


def wire_self(length, radius):
    """Self term of a straight round wire, ``Lp = (mu0 l/2pi)[ln(2l/a) - 3/4]``, including the
    internal inductance of a uniform current. Valid for ``l >> a``."""
    return 2.0 * (jnp.log(2.0 * length / radius) - 0.75) / length


def pair_quadratic(pos, mom, g: Callable, self_g, group=None, chunk: int = 128):
    """``Σ_a Σ_b (mom_a · mom_b) g(|r_a − r_b|)`` over arbitrary element positions.

    Args:
        pos: ``(N, 3)`` element positions.
        mom: ``(N,)`` scalar or ``(N, d)`` vector density per element (for PEEC, ``J · volume``).
        g: the kernel, a callable of scalar distance.
        self_g: diagonal values ``g_aa`` — per ROW when ``group`` is None, per GROUP otherwise.
            Required; see the module docstring.
        group: optional ``(N,)`` integer element label. Rows sharing a label are one element made of
            several quadrature points: pairs *inside* an element are excluded and replaced by that
            element's analytic self term, while pairs in *different* elements see the true sub-point
            distances. That is the standard near/far split, and it is how near-field accuracy is
            bought — a single point per element is 7.8 % low on collinear neighbours, falling to
            2.5 % at two Gauss points and 0.2 % at eight. ``None`` means one point per element,
            where the group test degenerates to the diagonal.
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
    if group is None:
        grp_np = None
        gsel = jnp.arange(n)
    else:
        import numpy as _np

        grp_np = _np.asarray(group)
        gsel = jnp.asarray(grp_np)

    P = jnp.zeros((npad, pos.shape[1]), pos.dtype).at[:n].set(pos)
    M = jnp.zeros((npad, mom.shape[1]), mom.dtype).at[:n].set(mom)
    # Pads take label -1 so pad-pad is excluded by the same test; pad-real is caught by `real`.
    G = jnp.full((npad,), -1).at[:n].set(gsel)
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
        excl = (G[idx][:, None] == G[None, :]) | (~real)[None, :] | (~real[idx])[:, None]
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
    if grp_np is None:
        return acc + jnp.sum((mom * mom).sum(1) * jnp.asarray(self_g))
    # One element's moment is the sum over its quadrature points, so the self term acts on that
    # total rather than on each point -- otherwise the element's own extent is counted twice.
    ne = int(grp_np.max()) + 1
    Me = jax.ops.segment_sum(mom, jnp.asarray(grp_np), num_segments=ne)
    return acc + jnp.sum((Me * Me).sum(1) * jnp.asarray(self_g))


def lattice_kernel(n: Sequence[int], h: Sequence[float], g: Callable, self_g, sub=None, w=None):
    """The BTTB generator on the doubled grid, ready for :func:`jax.numpy.fft.rfftn`.

    On a regular lattice ``G(i, j) = g(i − j)``, so the operator is block-Toeplitz Toeplitz-block and
    the number of independent coefficients falls from O(N²) to O(N). Embedding in a circulant of
    twice the size per axis turns the apply into a pointwise product in Fourier space.

    ``sub``/``w`` give per-element quadrature points and their moment weights; with them the
    generator is the full double sum over sub-points, which is what makes the FFT path agree with the
    dense operator instead of approximating it. Omit them for the one-point rule.

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
    sep = [o * hi for o, hi in zip(off, h)]  # the displacement between two lattice cells
    if sub is None:
        d = jnp.sqrt(sum(v * v for v in sep))
        body = jnp.where(d > 0, g(jnp.where(d > 0, d, 1.0)), self_g)
        return jnp.where(valid, body, 0.0)

    # Sub-point quadrature, and it stays Toeplitz: every element of a family carries the SAME
    # sub-point offsets, so the double sum still depends only on the cell separation. Without it the
    # lattice operator is the one-point rule, which on a bar lattice is 4.2 % low against a converged
    # quadrature -- too much for a path whose whole claim is being the exact fast alternative.
    sub = jnp.asarray(sub)
    w = jnp.ones(sub.shape[0]) if w is None else jnp.asarray(w)
    body = jnp.zeros(off[0].shape)
    for a in range(sub.shape[0]):
        for b in range(sub.shape[0]):
            ds = sub[a] - sub[b]
            d = jnp.sqrt(sum((v + ds[i]) ** 2 for i, v in enumerate(sep)))
            body = body + w[a] * w[b] * jnp.where(d > 0, g(jnp.where(d > 0, d, 1.0)), 0.0)
    at0 = tuple(0 for _ in n)  # the element's own term is analytic, not a quadrature of a singularity
    body = body.at[at0].set(self_g * jnp.sum(w) ** 2)
    return jnp.where(valid, body, 0.0)


def lattice_operator(n: Sequence[int], h: Sequence[float], g: Callable, self_g, sub=None, w=None):
    """Return ``apply(x)`` performing the BTTB matvec by FFT.

    ``x`` has the lattice shape ``n``. The generator's transform is computed once here and closed
    over, so a Krylov solve pays for it a single time rather than per iteration.
    """
    n = tuple(int(v) for v in n)
    dbl = [2 * v for v in n]
    ghat = jnp.fft.rfftn(lattice_kernel(n, h, g, self_g, sub=sub, w=w))

    def apply(x):
        xp = jnp.zeros(dbl, x.dtype).at[tuple(slice(0, v) for v in n)].set(x)
        y = jnp.fft.irfftn(ghat * jnp.fft.rfftn(xp), s=dbl)
        return y[tuple(slice(0, v) for v in n)]

    return apply


def pair_matrix(pos, mom, g: Callable, self_g, group=None):
    """The element-by-element operator itself, ``K_ab``, rather than the scalar ``x'Kx``.

    :func:`pair_quadratic` contracts the whole double sum; a circuit solve needs the matrix, and so
    does any per-element readout -- which segment to widen is PEEC's one genuine advantage over a
    field method, and it lives in the off-diagonal.

    Dense and O(N_element^2) in memory, deliberately: this is the small-network path. Beyond a few
    thousand elements use :func:`pair_quadratic` for energies and :func:`lattice_operator` for
    applies, neither of which forms the matrix.
    """
    if self_g is None:
        raise ValueError("pair_matrix: self_g is required, for the same reason pair_quadratic requires it.")
    pos = jnp.asarray(pos)
    mom = jnp.asarray(mom)
    if mom.ndim == 1:
        mom = mom[:, None]
    n = pos.shape[0]
    grp = jnp.arange(n) if group is None else jnp.asarray(group)
    ne = int(jnp.max(grp)) + 1

    d = pos[:, None, :] - pos[None, :, :]
    r = jnp.sqrt(jnp.clip((d * d).sum(-1), 0.0))
    same = grp[:, None] == grp[None, :]
    # substitute before the kernel, then zero the contribution: g may be singular at 0, and masking
    # afterwards would still differentiate the dead branch (0 * inf = NaN in reverse mode).
    kk = jnp.where(same, 0.0, g(jnp.where(same, 1.0, r)))
    sub = (mom @ mom.T) * kk
    # contract sub-points into their elements
    blk = jax.ops.segment_sum(sub, grp, num_segments=ne)
    blk = jax.ops.segment_sum(blk.T, grp, num_segments=ne).T
    Me = jax.ops.segment_sum(mom, grp, num_segments=ne)
    return blk + jnp.diag((Me * Me).sum(1) * jnp.asarray(self_g))
