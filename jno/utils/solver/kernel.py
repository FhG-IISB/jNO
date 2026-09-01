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
import numpy as np

__all__ = [
    "pair_quadratic",
    "pair_matrix",
    "lattice_kernel",
    "lattice_operator",
    "sphere_self",
    "bar_self",
    "wire_self",
    "internal_impedance",
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


def _xp(*vals):
    """``numpy`` when every input is concrete, ``jax.numpy`` when any is traced.

    A self term is a closed form in the element's own dimensions, so it follows whatever those are:
    a fixed lattice keeps it a HOST constant -- which is what lets the discretisation run inside
    ``jax.jit``, where a ``jnp`` result would be staged into the jaxpr and could no longer be read
    back as a float -- while a traced radius (a bond wire's gauge as a design variable) keeps it in
    ``jax`` so the gradient reaches it.
    """
    return jnp if any(isinstance(v, jax.core.Tracer) for v in vals) else np


#: Gauss-Legendre order for :func:`bar_self`. The transformed integrand is a low-order polynomial
#: over ``sqrt(w^2 + t^2 u^2 + l^2 v^2)``, so the order needed is set by how far apart the three
#: dimensions are: at a lattice's own aspect ratios 24 suffices, but a 10,000:1 sliver needs more.
#: Measured against order 160 -- 24: -23 % at l/w = 1e4; 48: -9.5 %; 96: +0.003 %. A lattice has one
#: shape per axis family, so this runs a handful of times per build and the cost does not matter.
_SELF_ORDER = 96

#: Unique shapes evaluated per pass, bounding the transient at ``chunk * order^3`` doubles.
_SELF_CHUNK = 4


def _self_integral(w, t, ell):
    """``int_0^w int_0^t int_0^l (w-x)(t-y)(l-z) / r dx dy dz`` -- exact, at any aspect ratio.

    Substituting ``x = w xi`` etc. and splitting the unit cube by which coordinate is largest, the
    Duffy map ``xi = s, eta = s u, zeta = s v`` carries a Jacobian ``s^2`` that cancels the ``1/r``
    exactly, leaving

        s (1-s) (1-s u) (1-s v) / sqrt(w^2 + t^2 u^2 + l^2 v^2)

    which is smooth on ``[0,1]^3``. The other two regions are the same with the roles permuted.
    """
    x, wgt = np.polynomial.legendre.leggauss(_SELF_ORDER)
    x, wgt = 0.5 * (x + 1.0), 0.5 * wgt  # map to [0, 1]
    s = x[:, None, None]
    u = x[None, :, None]
    v = x[None, None, :]
    W = wgt[:, None, None] * wgt[None, :, None] * wgt[None, None, :]
    w, t, ell = np.asarray(w, float), np.asarray(t, float), np.asarray(ell, float)
    out = np.zeros(np.broadcast(w, t, ell).shape, dtype=float)
    # each region is one coordinate playing the role of `s`; the scale under the root follows it
    for A, B, C in ((w, t, ell), (t, w, ell), (ell, w, t)):
        A, B, C = A[..., None, None, None], B[..., None, None, None], C[..., None, None, None]
        num = s * (1.0 - s) * (1.0 - s * u) * (1.0 - s * v)
        den = np.sqrt(A**2 + (B * u) ** 2 + (C * v) ** 2)
        out = out + np.sum(W * num / den, axis=(-3, -2, -1))
    return (np.asarray(w) * np.asarray(t) * np.asarray(ell)) ** 2 * out


def bar_self(length, width, thickness):
    """Self term of a straight rectangular bar, EXACT at any aspect ratio.

    The defining double-volume integral,

        Lp = (mu0 / 4 pi A^2) int_V int_V dV dV' / |r - r'|,    A = w t

    reduced by the difference variable to ``8 W(w, t, l)`` and evaluated by :func:`_self_integral`.
    Returned as the ``g_aa`` this module wants, ``Lp = (mu0/4pi) l^2 g_aa``.

    This replaces Ruehli's asymptotic form (*IBM J. Res. Dev.* 16:470, 1972, eq. 12),
    ``(mu0 l/2pi)[ln(2l/(w+t)) + 0.5 + 0.2235 (w+t)/l]``, which is **valid only for l >> w, t** --
    a condition a lattice violates as a matter of course. One cell through a 0.5 mm conductor at a
    0.06 mm lateral pitch is an element eight times shorter than it is thick, and there
    ``ln(2l/(w+t))`` is negative. It did not fail loudly: on the copper bar of Romano et al. (IEEE
    TEMC 65(2), 2023, sec. V-A), where Q3D and three PEEC variants agree at 2.85 nH, refining the
    lattice laterally gave 3.01 -> 3.23 -> 3.41 nH -- moving AWAY from the answer, which reads as a
    mesh that has not converged rather than as a broken kernel.

    A host constant of the grid: a lattice has one shape per axis family, so this is evaluated a
    handful of times per build and the quadrature cost is irrelevant.
    """
    length, width, thickness = np.asarray(length, float), np.asarray(width, float), np.asarray(thickness, float)
    # one shape per axis family in practice, so solve the distinct ones and scatter back
    key = np.stack(np.broadcast_arrays(length, width, thickness), axis=-1).reshape(-1, 3)
    uniq, inv = np.unique(np.round(key, 15), axis=0, return_inverse=True)
    parts = []
    for k in range(0, len(uniq), _SELF_CHUNK):
        u = uniq[k : k + _SELF_CHUNK]
        parts.append(8.0 * _self_integral(u[:, 1], u[:, 2], u[:, 0]) / (u[:, 0] ** 2 * u[:, 1] ** 2 * u[:, 2] ** 2))
    g = np.concatenate(parts)
    return g[inv].reshape(np.broadcast(length, width, thickness).shape)


def wire_self(length, radius):
    """Self term of a straight round wire, ``Lp = (mu0 l/2pi)[ln(2l/a) - 3/4]``, including the
    internal inductance of a uniform current. Valid for ``l >> a``."""
    return 2.0 * (_xp(length, radius).log(2.0 * length / radius) - 0.75) / length


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


def lattice_kernel(n: Sequence[int], h: Sequence[float], g: Callable, self_g, sub=None, w=None, sub_b=None, w_b=None):
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
    # `sub_b` makes this the CROSS generator between two families on the same grid -- the two current
    # sheets of one slab, offset to its opposite faces. Then no pair of elements ever coincides: the
    # two sheets of a cell are a real thickness apart, so the quadrature is regular and there is no
    # self term to substitute. Getting that wrong would put an analytic self inductance where a
    # perfectly ordinary mutual belongs.
    cross = sub_b is not None
    sb = sub if not cross else jnp.asarray(sub_b)
    wb = w if not cross else (jnp.ones(sb.shape[0]) if w_b is None else jnp.asarray(w_b))
    body = jnp.zeros(off[0].shape)
    for a in range(sub.shape[0]):
        for b in range(sb.shape[0]):
            ds = sub[a] - sb[b]
            d = jnp.sqrt(sum((v + ds[i]) ** 2 for i, v in enumerate(sep)))
            body = body + w[a] * wb[b] * jnp.where(d > 0, g(jnp.where(d > 0, d, 1.0)), 0.0)
    if not cross:
        at0 = tuple(0 for _ in n)  # its own term is analytic, not a quadrature of a singularity
        body = body.at[at0].set(self_g * jnp.sum(w) ** 2)
    return jnp.where(valid, body, 0.0)


def lattice_operator(n: Sequence[int], h: Sequence[float], g: Callable, self_g, sub=None, w=None, sub_b=None, w_b=None, transpose=False):
    """Return ``apply(x)`` performing the BTTB matvec by FFT.

    ``x`` has the lattice shape ``n``. The generator's transform is computed once here and closed
    over, so a Krylov solve pays for it a single time rather than per iteration.
    """
    n = tuple(int(v) for v in n)
    dbl = [2 * v for v in n]
    ghat = jnp.fft.rfftn(lattice_kernel(n, h, g, self_g, sub=sub, w=w, sub_b=sub_b, w_b=w_b))
    # A cross generator is not even in the separation -- the offset between the two families has a
    # sign -- so the block is not symmetric and its transpose is a distinct operator. Reversing the
    # generator is conjugation in Fourier space, which is the whole cost of getting `K_BA` from
    # `K_AB`; the FULL 2x2 block operator is symmetric, as a partial inductance must be.
    if transpose:
        ghat = jnp.conjugate(ghat)

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
    # `ne` is STRUCTURAL -- how many elements the sub-points group into -- so it is read from the
    # host array, never back out of jnp. Inside a jit even a constant stages into the jaxpr, and
    # `int()` on the result raises; going through numpy is what keeps this callable under jit.
    ne = n if group is None else int(np.asarray(group).max()) + 1

    d = pos[:, None, :] - pos[None, :, :]
    same = grp[:, None] == grp[None, :]
    # Substitute BEFORE the sqrt, not just before the kernel. A same-element pair sits at zero
    # separation, d(sqrt)/dx is infinite there, and `where` afterwards differentiates the dead branch
    # anyway -- so the masked-out entries poisoned the whole gradient with NaN while the VALUE was
    # perfectly correct. Differentiating w.r.t. the positions themselves is what exposed it; nothing
    # about the forward operator ever showed it. Same guard as pair_quadratic already carries.
    r = jnp.sqrt(jnp.where(same, 1.0, (d * d).sum(-1)))
    # Zero the CONTRIBUTION rather than sending r to infinity: g(inf) -> 0 happens to hold for 1/r,
    # but a caller-supplied kernel carries no such promise.
    kk = jnp.where(same, 0.0, g(r))
    sub = (mom @ mom.T) * kk
    # contract sub-points into their elements
    blk = jax.ops.segment_sum(sub, grp, num_segments=ne)
    blk = jax.ops.segment_sum(blk.T, grp, num_segments=ne).T
    Me = jax.ops.segment_sum(mom, grp, num_segments=ne)
    return blk + jnp.diag((Me * Me).sum(1) * jnp.asarray(self_g))


def internal_impedance(length, area, skin, round_, omega, sigma, mu0=4e-7 * jnp.pi, span=1):
    """Per-element internal impedance, replacing the DC resistance with a shape-aware surface one.

    An element's own resistance is not ``rho l / A`` once its transverse size approaches the skin
    depth: the current retreats to the surface and both the resistance and the internal inductance
    change. The classical closed forms, with ``gamma = sqrt(j w mu sigma)``:

        round wire, radius a     Z = (gamma rho / 2 pi a)  I0(gamma a) / I1(gamma a)
        slab, thickness t        Z = (gamma rho / 2 w)     coth(gamma t / 2)

    both reducing to ``rho l / A`` as ``w -> 0``. See Ramo, Whinnery & Van Duzer, *Fields and Waves
    in Communication Electronics*, 3rd ed., sec. 3.16-3.17.

    This is what lets a conductor be ONE element through its thickness rather than several, which is
    how the PEEC literature keeps a package tractable (Romano, Kovacevic-Badstuebner, Antonini &
    Grossner, *Efficient PEEC iterative solver for power electronic applications*, IEEE Trans.
    Electromagn. Compat. 65(2), 2023, sec. II-A). It also strengthens the preconditioner, whose
    diagonal is ``Z_s + j w Lp_aa``: at a megahertz ``Z_s`` is many times the DC value, so the
    diagonal carries much more of the true operator.

    A SHAPE-AWARE coefficient rather than a flat Leontovich one because the difference is not small:
    on a bond wire the plane-surface approximation is 12.5 % out where the cylindrical form is 0.02 %
    at any ``a / delta``.

    ``span`` -- THE CONDITION THIS FORM DEPENDS ON. Both closed forms describe a conductor whose
    surfaces are free, so an element may take one only when it IS the whole conductor across its
    thickness. Stack two elements through a thickness and each still counts the shared interface as
    an exposed surface, so the pair conducts about twice as well as the single element it replaced.
    Measured on a 40 x 4 x 2 mm bar at 1 MHz, splitting only the thickness:

        cells through thickness      1        2        4
        R, if every element takes    1239     620      8717     uOhm
        the surface form             x1.00    x0.50    x7.03

    ``span`` is the number of elements across the thickness, so ``span == 1`` is the case the forms
    describe. Anywhere else the element takes ``rho l / A`` instead. That is not a fallback but
    the right model for a subdivided conductor: the elements then resolve the current distribution
    themselves, which is what makes them worth having, and the surface impedance exists precisely to
    avoid needing them. The default is ``True`` because a lone conductor asked about on its own is
    the whole of itself; a lattice passes its own per-element flag.
    """
    length = jnp.asarray(length)
    area = jnp.asarray(area)
    skin = jnp.asarray(skin)
    rho = 1.0 / jnp.asarray(sigma)
    w = jnp.asarray(omega)
    dc = rho * length / area
    # substitute BEFORE the sqrt: d(sqrt)/dx is infinite at 0, so differentiating a masked branch
    # that evaluated sqrt(0) gives NaN even though the value is discarded
    dead = w == 0
    g = jnp.sqrt(jnp.where(dead, 1.0, 1j * w * mu0 / rho))

    # round wire: I0/I1 by its ratio, which stays finite as gamma a -> 0 (where it behaves as 2/(g a))
    ga = g * skin
    ratio = _i0_over_i1(ga)
    z_round = (g * rho / (2.0 * jnp.pi * skin)) * ratio * length

    # slab: width from the area, thickness `skin`; coth is 1/x as x -> 0
    wid = area / skin
    gt = 0.5 * g * skin
    z_slab = (
        (g * rho / (2.0 * wid))
        * jnp.where(jnp.abs(gt) < 1e-6, 1.0 / jnp.where(jnp.abs(gt) < 1e-6, 1.0, gt), 1.0 / jnp.tanh(gt))
        * length
    )

    z = jnp.where(jnp.asarray(round_), z_round, z_slab)
    z = jnp.where(jnp.asarray(span) == 1, z, dc.astype(z.dtype))  # subdivided: no free surface to speak of
    return jnp.where(dead, dc.astype(z.dtype), z)


def slab_transfer_impedance(length, width, thickness, omega, sigma, mu0=4e-7 * jnp.pi):
    """The 2-port form of the slab impedance: one current sheet per face, and their coupling.

    :func:`internal_impedance` gives a slab ONE impedance, which forces its current to be a single
    unknown. That is what makes the element's inductance disagree with its resistance: the resistance
    knows the current is confined to a skin layer at the faces, but with one unknown there is nowhere
    to put it except spread through the whole cell. A return plane's THICKNESS then changes L at a
    frequency where the copper below the skin layer is invisible -- measured against pypeec 5.8.0,
    which resolves the skin depth with volume cells, +21.3 % where pypeec gives -0.05 %.

    Giving each face its own sheet current lets the SOLVE find the split rather than assuming it,
    which is the whole point: on a 1.6 mm plane at 7.7 skin depths the answer is about 92/8, and no
    fixed rule reproduces that (a 50/50 face split was measured 16 % out).

    The two sheets are the ports of the conducting slab treated as a transmission line across its
    own thickness (Ramo, Whinnery & Van Duzer, *Fields and Waves in Communication Electronics*,
    3rd ed., sec. 3.16-3.17)::

        [V1]   gamma rho l  [  coth(gamma t)   csch(gamma t) ] [I1]
        [V2] = -----------  [  csch(gamma t)   coth(gamma t) ] [I2]
                    w

    Returns ``(z_self, z_mutual)`` -- the diagonal and the off-diagonal.

    **It reduces exactly to the one-unknown model.** The sheets sit in parallel across the same pair
    of nodes, so equal currents ``I1 = I2 = I/2`` give ``V = (z_self + z_mutual) I / 2``, and since
    ``coth(x) + csch(x) = coth(x/2)`` that is ``(gamma rho l / 2w) coth(gamma t / 2)`` -- the slab
    branch of :func:`internal_impedance`, to the last bit. So the ``delta >> t`` limit is not merely
    close, it is the same number, which is what makes this safe to switch on.

    At DC both entries are ``rho l / A``: the pair is then degenerate and the split is decided by the
    partial inductance rather than here, which is why a sheet pair is only ever emitted where the
    conductor is thick against the skin depth (there is no skin depth at all at DC).
    """
    rho = 1.0 / jnp.asarray(sigma)
    length, width, thickness = jnp.asarray(length), jnp.asarray(width), jnp.asarray(thickness)
    w = jnp.asarray(omega)
    dc = rho * length / (width * thickness)
    # as in `internal_impedance`: substitute BEFORE the sqrt, or differentiating the masked branch
    # through sqrt(0) gives NaN even where the value is discarded
    dead = w == 0
    g = jnp.sqrt(jnp.where(dead, 1.0, 1j * w * mu0 / rho))
    gt = g * thickness
    # coth and csch both go as 1/gt as gt -> 0, and `pre / gt` is exactly `dc` there, so the small
    # branch needs no series: it is the DC resistance on the nose.
    small = jnp.abs(gt) < 1e-6
    safe = jnp.where(small, 1.0, gt)
    pre = g * rho * length / width
    z_self = jnp.where(small, dc.astype(complex), pre / jnp.tanh(safe))
    z_mut = jnp.where(small, dc.astype(complex), pre / jnp.sinh(safe))
    return (jnp.where(dead, dc.astype(complex), z_self), jnp.where(dead, dc.astype(complex), z_mut))


def _i0_over_i1(z):
    """``I0(z) / I1(z)`` for complex ``z``, over the whole range a conductor reaches.

    Three regimes, because no single form covers them:

    * ``|z| -> 0``   the ratio is ``2/z``, which is what makes the wire formula reduce to ``rho l/A``
    * ``|z| < 12``   the ascending series, which converges quickly there
    * ``|z| > 12``   the ASYMPTOTIC series. The ascending one fails catastrophically here: at 10 MHz a
      1 mm copper wire has ``|z| ~ 68``, whose terms reach ``e^68`` and cancel to nothing in double
      precision. It read 70 % high against the thin-skin limit before this branch existed.

    Abramowitz & Stegun, *Handbook of Mathematical Functions*, 9.6.12 and 9.7.1.
    """
    z = jnp.asarray(z)
    a = jnp.abs(z)
    tiny, big = a < 1e-6, a > 12.0
    zs = jnp.where(tiny | big, 1.0, z)  # a safe argument for the branch that must not overflow
    k = jnp.arange(0, 40)
    i0 = ((0.25 * zs[..., None] ** 2) ** k / jnp.exp(2.0 * jax.scipy.special.gammaln(k + 1.0))).sum(-1)
    i1 = (
        (0.5 * zs[..., None]) ** (2 * k + 1)
        / jnp.exp(jax.scipy.special.gammaln(k + 1.0) + jax.scipy.special.gammaln(k + 2.0))
    ).sum(-1)
    zb = jnp.where(big, z, 1.0)
    a0 = 1.0 + 1.0 / (8 * zb) + 9.0 / (128 * zb**2) + 225.0 / (3072 * zb**3)
    a1 = 1.0 - 3.0 / (8 * zb) - 15.0 / (128 * zb**2) - 105.0 / (3072 * zb**3)
    return jnp.where(tiny, 2.0 / jnp.where(tiny, 1.0, z), jnp.where(big, a0 / a1, i0 / i1))
