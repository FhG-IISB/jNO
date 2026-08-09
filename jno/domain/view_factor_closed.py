"""Closed-form axisymmetric view factors: the azimuthal integral is evaluated analytically.

The rotational integral of the diffuse kernel has a primitive in closed form, so the 2*pi sweep never
has to be quadratured and no ``n_phi`` parameter exists. Occlusion boundaries are likewise algebraic
rather than sampled, so a shadow edge is resolved exactly instead of to an azimuthal bin width.

Reference: F. Dupret, P. Nicodeme, Y. Ryckmans, P. Wouters and M. J. Crochet, "Global modelling of
heat transfer in crystal growth furnaces", *Int. J. Heat Mass Transfer* **33**(9) 1849-1871 (1990),
section 3.2. The paper states the form of its eq. (24) but not its coefficients; they are derived
below.

Why this replaces the sampled kernel
------------------------------------
The previous path sphere-traced every element pair at every azimuth against a rasterised signed
distance field whose cell size is set by the *thinnest solid* in the scene. On the reference furnace
that made the 189-element growth cavity cost twice as much as the 477-element chamber. Measured on
that furnace: 106.3 s -> 9.0 s (11.8x), with closure error simultaneously *lower* than the kernel it
replaces (2.8e-02 against 9.2e-02 on the cavity, 3.4e-02 against 4.2e-02 on the chamber, both before
any ``enforce_closure`` normalisation).

The mathematics
---------------
With ``x = (r,0,z)`` carrying unit normal ``(n_r, n_z)``, ``x' = (r',0,z')`` carrying ``(n_r', n_z')``,
``d = z' - z``, and ``x'`` rotated to azimuth ``t``, every factor of the Lambert kernel is a rational
function of ``cos t`` alone::

    (x*-x).n   = a'  + b'  cos t      a'  =  d n_z  - r  n_r ,   b'  = r' n_r
    (x-x*).n*  = a'' + b'' cos t      a'' = -r' n_r'- d  n_z',   b'' = r  n_r'
    |x*-x|^2   = a   + b   cos t      a   = r^2 + r'^2 + d^2 ,   b   = -2 r r'

so with ``s = sqrt(a^2-b^2)``, ``S(t) = sin t/(a + b cos t)`` and
``T(t) = (2/s) atan2(sqrt(a-b) sin(t/2), sqrt(a+b) cos(t/2))``::

    G0(t) = (a T - b S)/s^2      G1(t) = (a S - b T)/s^2      G2(t) = [t - 2a T + a^2 G0]/b^2

    integral over [t0,t1] = a'a'' G0 + (a'b'' + a''b') G1 + b'b'' G2      (evaluated t1 - t0)

``G0`` and ``G1`` follow from solving the 2x2 system formed by
``d/dt[sin t/(a+b cos t)] = (a cos t + b)/(a+b cos t)^2`` and
``(a + b cos t)/(a+b cos t)^2 = 1/(a+b cos t)``. ``T`` uses ``atan2`` rather than
``atan(tan(t/2))``, which would blow up at ``t = pi``.

Numerical hazards, all of which produce plausible wrong answers rather than crashes
----------------------------------------------------------------------------------
* **Coincident points** (``r = r'``, ``d = 0`` -- the diagonal's ``q == p`` term) give ``a + b = 0``,
  so ``s = 0`` and every ``G`` diverges *individually* while their combination stays finite: numerator
  and denominator share the factor ``(1 - cos t)^2`` exactly and the kernel is the constant
  ``b'b''/b^2``. A ring genuinely sees itself around the azimuth, so this term is physical.
* **Near-axis** ``b = -2 r r' -> 0`` leaves a removable ``1/b^2`` in ``G2``, switched to its limit.
* **Grazing chords** make the intersection quadratic tangent, discriminant exactly zero -- and rounding
  lands it on either side (measured -16 against terms of order 7.6e16). Testing ``disc >= 0`` discards
  the root and can zero an entire pair.
* **Horizontal or vertical sides** have zero extent along one axis, so an on-segment window must scale
  with the side, not be an absolute epsilon.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["view_factor_axisymmetric_closed", "segments_from_polygons"]

_TINY = 1e-300


# ---------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------
def segments_from_polygons(geoms: Sequence) -> np.ndarray:
    """Flatten shapely meridian polygons to ``(S, 4)`` occluding segments ``(r_j, z_j, r_k, z_k)``.

    Interior rings are included as well as exteriors: a "leftover void" region built as
    ``scene.difference(union(solids))`` is a polygon **with holes**, and dropping its interiors
    silently removes most of the occluders.
    """
    out = []
    for poly in geoms or ():
        for g in getattr(poly, "geoms", [poly]):
            ext = getattr(g, "exterior", None)
            if ext is None:
                continue
            for ring in [ext, *list(getattr(g, "interiors", []))]:
                xy = np.asarray(ring.coords, dtype=float)
                for k in range(len(xy) - 1):
                    out.append((xy[k, 0], xy[k, 1], xy[k + 1, 0], xy[k + 1, 1]))
    return np.asarray(out, dtype=float) if out else np.zeros((0, 4))


# ---------------------------------------------------------------------------------------
# the azimuthal primitive
# ---------------------------------------------------------------------------------------
def _prim(ap, bp, app, bpp, a, b, t):
    """Azimuthal integral of the kernel from 0 to ``t``; zero at ``t = 0`` by construction."""
    degen = (a + b) <= 1e-13 * np.maximum(np.abs(a), _TINY)
    # The degenerate branch is computed and then discarded by np.where, so its overflow is expected
    # and meaningless; silencing it here keeps a real warning elsewhere visible.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        s = np.sqrt(np.maximum(a * a - b * b, _TINY))
        T = (2.0 / s) * np.arctan2(
            np.sqrt(np.maximum(a - b, 0.0)) * np.sin(t / 2.0), np.sqrt(np.maximum(a + b, 0.0)) * np.cos(t / 2.0)
        )
        den = a + b * np.cos(t)
        S = np.sin(t) / np.where(np.abs(den) < _TINY, _TINY, den)
        s2 = s * s
        G0 = (a * T - b * S) / s2
        G1 = (a * S - b * T) / s2
        small = np.abs(b) < 1e-7 * np.maximum(np.abs(a), _TINY)
        G2 = np.where(
            small,
            (t / 2.0 + np.sin(2.0 * t) / 4.0) / np.maximum(a * a, _TINY),
            (t - 2.0 * a * T + a * a * G0) / np.where(small, 1.0, b * b),
        )
        val = ap * app * G0 + (ap * bpp + app * bp) * G1 + bp * bpp * G2
        b2 = np.where(np.abs(b) < _TINY, 1.0, b * b)
        val = np.where(degen, bp * bpp / b2 * t, val)  # coincident-point limit
    return np.where(t <= 0.0, 0.0, val)


def _coeffs(r, z, nr, nz, rp, zp, nrp, nzp):
    d = zp - z
    return (d * nz - r * nr, rp * nr, -rp * nrp - d * nzp, r * nrp, r * r + rp * rp + d * d, -2.0 * r * rp)


def _facing_bounds(alpha, beta):
    """``[lo, hi]`` within ``[0, pi]`` where ``alpha + beta cos t > 0``; empty as ``lo == hi == 0``.

    Each cosine factor is linear in ``cos t`` and ``cos`` is monotone on ``[0, pi]``, so this is
    exactly one interval. Omitting it lets the back-facing range contribute *negative* kernel that
    cancels the front-facing part -- a silent error, not a small one.
    """
    lo = np.zeros_like(alpha)
    hi = np.full_like(alpha, np.pi)
    flat = np.abs(beta) < _TINY
    safe = np.where(flat, 1.0, beta)
    x = -alpha / safe
    ac = np.arccos(np.clip(x, -1.0, 1.0))
    pos = (~flat) & (beta > 0)
    neg = (~flat) & (beta < 0)
    hi = np.where(pos, ac, hi)
    lo = np.where(neg, ac, lo)
    dead = (flat & (alpha <= 0)) | (pos & (x >= 1.0)) | (neg & (x <= -1.0))
    return np.where(dead, 0.0, lo), np.where(dead, 0.0, hi)


# ---------------------------------------------------------------------------------------
# occlusion
# ---------------------------------------------------------------------------------------
def _blocked_at(r, z, rp, zp, c, side):
    """Exact predicate: is the chord blocked by ``side`` at ``cos(theta) = c``?

    The chord's meridian projection is ``rho(t)^2 = p0 + p1 t + p2 t^2``; intersecting it with the
    side's line squares to a quadratic in ``t``.
    """
    rj, zj, rk, zk = side
    al, be = (zk - zj), -(rk - rj)
    ga = -(al * rj + be * zj)
    dz = zp - z
    p0 = r * r
    p1 = 2.0 * r * (rp * c - r)
    p2 = r * r - 2.0 * r * rp * c + rp * rp
    q0, q1 = be * z + ga, be * dz
    A = al * al * p2 - q1 * q1
    B = al * al * p1 - 2.0 * q0 * q1
    C = al * al * p0 - q0 * q0
    disc = B * B - 4.0 * A * C
    dscale = np.maximum(B * B, np.abs(4.0 * A * C))
    degen = disc <= 1e-9 * dscale  # tangent / plane-crossing: one root, not two
    ok = disc >= -1e-9 * dscale
    sq = np.where(degen, 0.0, np.sqrt(np.maximum(disc, 0.0)))
    out = np.zeros(np.broadcast(r, z, rp, zp, c).shape, dtype=bool)
    lin = np.abs(A) <= _TINY
    sl = float(np.hypot(rk - rj, zk - zj))
    tol = 1e-9 * max(sl, 1e-9)  # scaled: a horizontal side has zero z-extent
    for sgn in (1.0, -1.0):
        t = np.where(lin, -C / np.where(np.abs(B) > _TINY, B, 1.0), (-B + sgn * sq) / np.where(lin, 1.0, 2.0 * A))
        inside = ok & (t > 1e-9) & (t < 1.0 - 1e-9) & (~lin | (np.abs(B) > _TINY))
        zt = z + t * dz
        rho = np.sqrt(np.maximum(p0 + p1 * t + p2 * t * t, 0.0))
        sign_ok = (al * rho) * (-(q0 + q1 * t)) > -1e-9 * np.maximum(np.abs(al * rho), _TINY)
        on = (zt >= min(zj, zk) - tol) & (zt <= max(zj, zk) + tol) & (rho >= min(rj, rk) - tol) & (rho <= max(rj, rk) + tol)
        out |= inside & sign_ok & on
    return out


def _blocked_interval(r, z, rp, zp, side):
    """Blocked azimuth interval ``[t_lo, t_hi]`` for one side; empty as ``t_lo == t_hi == 0``.

    Candidate boundaries are algebraic -- a root at a vertex (the paper's eq. 31), a double root
    (eq. 32), and the endpoints -- so each sub-interval between consecutive candidates has constant
    state and one midpoint test settles it. No scanning, hence no sliver can be missed.
    """
    rj, zj, rk, zk = side
    dz = zp - z
    shape = np.broadcast(r, z, rp, zp).shape
    cand = [np.full(shape, -1.0), np.full(shape, 1.0)]

    for rv, zv in ((rj, zj), (rk, zk)):  # eq. 31
        den = 2.0 * r * rp * (zv - z) * (zp - zv)
        num = rv * rv * dz * dz - r * r * (zp - zv) ** 2 - rp * rp * (zv - z) ** 2
        cand.append(np.where(np.abs(den) > _TINY, num / np.where(np.abs(den) > _TINY, den, 1.0), np.nan))

    al, be = (zk - zj), -(rk - rj)
    ga = -(al * rj + be * zj)
    q0, q1 = be * z + ga, be * dz
    a2 = al * al
    A0, A1 = a2 * (r * r + rp * rp) - q1 * q1, -2.0 * a2 * r * rp
    B0, B1 = -2.0 * a2 * r * r - 2.0 * q0 * q1, 2.0 * a2 * r * rp
    C = a2 * r * r - q0 * q0
    qa, qb, qc = B1 * B1, 2.0 * B0 * B1 - 4.0 * C * A1, B0 * B0 - 4.0 * C * A0
    dd = qb * qb - 4.0 * qa * qc
    scale = np.maximum(qb * qb, np.abs(4.0 * qa * qc))
    good = (np.abs(qa) > _TINY) & (dd >= -1e-9 * scale)  # eq. 32; tolerant of exact tangency
    sd = np.sqrt(np.maximum(dd, 0.0))
    for sgn in (1.0, -1.0):
        cand.append(np.where(good, (-qb + sgn * sd) / np.where(good, 2.0 * qa, 1.0), np.nan))

    lin_a, lin_b = A0 + B0 + C, A1 + B1  # a root crossing the far endpoint
    cand.append(np.where(np.abs(lin_b) > _TINY, -lin_a / np.where(np.abs(lin_b) > _TINY, lin_b, 1.0), np.nan))

    Cs = np.stack(cand, axis=-1)
    Cs = np.where(np.isfinite(Cs) & (Cs >= -1.0) & (Cs <= 1.0), Cs, np.nan)
    Cs = np.sort(Cs, axis=-1)
    lo = np.zeros(shape)
    hi = np.zeros(shape)
    any_b = np.zeros(shape, dtype=bool)
    for k in range(Cs.shape[-1] - 1):
        c0, c1 = Cs[..., k], Cs[..., k + 1]
        valid = np.isfinite(c0) & np.isfinite(c1) & (c1 - c0 > 1e-14)
        bl = valid & _blocked_at(r, z, rp, zp, np.where(valid, 0.5 * (c0 + c1), 0.0), side)
        lo = np.where(bl & ~any_b, c0, np.where(bl, np.minimum(lo, c0), lo))
        hi = np.where(bl & ~any_b, c1, np.where(bl, np.maximum(hi, c1), hi))
        any_b |= bl
    return (
        np.where(any_b, np.arccos(np.clip(hi, -1.0, 1.0)), 0.0),
        np.where(any_b, np.arccos(np.clip(lo, -1.0, 1.0)), 0.0),
    )


def _visible_gaps(mids, sides, max_gaps=6, chunk=64):
    """Merged visible azimuth intervals per element pair, occlusion only.

    Only the upper triangle is computed and then mirrored. The chord ``i -> j`` at azimuth ``t`` *is*
    the chord ``j -> i``, so the visible set is symmetric -- but the root formulae are not bitwise
    symmetric under the swap, and residual splitting near degeneracies leaks straight into
    reciprocity. Mirroring makes it exact by construction, and halves the work.
    """
    m = len(mids)
    r_, z_ = mids[:, 0], mids[:, 1]
    G0 = np.zeros((m, m, max_gaps))
    G1 = np.zeros((m, m, max_gaps))
    n_over = 0
    for a in range(0, m, chunk):
        b = min(a + chunk, m)
        R, Z = r_[a:b, None], z_[a:b, None]
        RP, ZP = r_[None, a:], z_[None, a:]
        # Conservative bounds on where the chord can go, used to skip sides that cannot possibly
        # block this pair. z is exact: z(t) is linear, so the chord stays within [min, max] of the
        # endpoints. The radius bound is ONE-SIDED on purpose -- rho(t) is bounded ABOVE by
        # max(r, r'), but NOT below by min(r, r'): the chord bows toward the axis with azimuth
        # (rho(t,phi)^2 - r_proj(t)^2 = 2t(1-t) r r'(cos phi - 1) <= 0), so it can dip under both
        # endpoint radii. Using a lower bound here would silently discard real occlusions.
        #
        # The bounds are compared with a TOLERANCE, not exactly. Geometry routinely puts an element
        # midpoint exactly on a side's endpoint (an element sitting at the top of the wall it abuts),
        # and there the comparison fails by ~1e-17 of representation noise -- pruning away a side that
        # genuinely blocks. Measured: exactly one entry of the 477x477 chamber matrix went wrong that
        # way, the self-view of an element whose z equalled a side's upper endpoint. `_blocked_at`
        # already carries a scaled on-segment tolerance for the same reason; the prune must be at
        # least as permissive as the predicate it is short-cutting, or it changes the answer.
        z_lo = np.minimum(Z, ZP)
        z_hi = np.maximum(Z, ZP)
        r_hi = np.maximum(R, RP)
        starts, ends = [], []
        for side in sides:
            rj, zj, rk, zk = side
            tol = 1e-7 * max(abs(rj), abs(zj), abs(rk), abs(zk), 1e-7)
            cand = (max(zj, zk) + tol >= z_lo) & (min(zj, zk) - tol <= z_hi) & (min(rj, rk) - tol <= r_hi)
            tl = np.zeros(cand.shape)
            th = np.zeros(cand.shape)
            idx = np.nonzero(cand)
            if idx[0].size:
                # _blocked_interval broadcasts, so the candidate subset goes through as 1-D
                sl, sh = _blocked_interval(R[idx[0], 0], Z[idx[0], 0], RP[0, idx[1]], ZP[0, idx[1]], side)
                tl[idx] = sl
                th[idx] = sh
            starts.append(tl)
            ends.append(th)
        if starts:
            SS, EE = np.stack(starts, -1), np.stack(ends, -1)
        else:
            SS = np.zeros((b - a, m - a, 0))
            EE = SS
        live = EE > SS + 1e-15
        SS = np.where(live, SS, np.pi)  # park empties at pi: +1 then -1, zero width
        EE = np.where(live, EE, np.pi)
        ev_t = np.concatenate([SS, EE, np.full(SS.shape[:2] + (1,), np.pi)], axis=-1)
        ev_d = np.concatenate([np.ones_like(SS), -np.ones_like(EE), np.zeros(SS.shape[:2] + (1,))], axis=-1)
        order = np.argsort(ev_t, axis=-1)
        ts = np.take_along_axis(ev_t, order, -1)
        depth = np.cumsum(np.take_along_axis(ev_d, order, -1), axis=-1)
        zero = np.zeros(ts.shape[:2] + (1,))
        prev = np.concatenate([zero, ts[..., :-1]], axis=-1)
        pdep = np.concatenate([zero, depth[..., :-1]], axis=-1)
        openg = (pdep <= 0) & (ts > prev + 1e-15)
        idx = np.cumsum(openg, axis=-1) - 1
        n_over += int((openg.sum(-1) > max_gaps).sum())
        for g in range(max_gaps):
            sel = openg & (idx == g)
            G0[a:b, a:, g] = np.where(sel, prev, 0.0).sum(-1)
            G1[a:b, a:, g] = np.where(sel, ts, 0.0).sum(-1)
    if n_over:
        import logging

        logging.getLogger(__name__).warning(
            "%d element pairs needed more than max_gaps=%d visible azimuth intervals; "
            "the excess was dropped. Raise max_gaps for strongly multiply-occluded geometry.",
            n_over,
            max_gaps,
        )
    up = np.triu(np.ones((m, m), bool))
    return (np.where(up[:, :, None], G0, G0.transpose(1, 0, 2)), np.where(up[:, :, None], G1, G1.transpose(1, 0, 2)))


# ---------------------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------------------
def view_factor_axisymmetric_closed(
    E0, E1, Nrm, *, occluders=(), n_quad: int = 3, max_gaps: int = 6, chunk: int = 64
) -> np.ndarray:
    """Element-to-element axisymmetric view factors, azimuth integrated in closed form.

    ``E0``/``E1`` are ``(m, 2)`` meridional endpoints ``(r, z)``; ``Nrm`` is ``(m, 2)`` unit normals
    pointing into the enclosure; ``occluders`` is an ``(S, 4)`` array of blocking segments (use
    :func:`segments_from_polygons`). ``n_quad`` is the meridional Gauss order, used on **both** sides
    of every pair so that ``A_i F_ij`` is manifestly symmetric and reciprocity holds to round-off.

    Returns ``(m, m)`` with ``F[i,j]`` the fraction of diffuse energy leaving element ``i`` that
    reaches ``j`` -- the same normalisation as the sampled kernel, ``sum_q a_q sum_p a_p K_c / sum_q
    a_q`` with ``a = r * w``. The diagonal is kept: a ring sees itself around the azimuth.
    """
    E0 = np.asarray(E0, dtype=float)
    E1 = np.asarray(E1, dtype=float)
    Nrm = np.asarray(Nrm, dtype=float)
    sides = np.asarray(occluders, dtype=float) if len(occluders) else np.zeros((0, 4))
    m = len(E0)
    gx, gw = np.polynomial.legendre.leggauss(int(n_quad))
    s = 0.5 * (gx + 1.0)
    L = np.linalg.norm(E1 - E0, axis=1)
    qp = E0[:, None, :] + s[None, :, None] * (E1 - E0)[:, None, :]
    w = (0.5 * gw)[None, :] * L[:, None]
    aq = qp[..., 0] * w  # ring area / 2*pi

    Gs, Ge = _visible_gaps(0.5 * (E0 + E1), sides, max_gaps=max_gaps, chunk=chunk)

    num = np.zeros((m, m))
    for qi in range(int(n_quad)):
        r_, z_ = qp[:, qi, 0][:, None], qp[:, qi, 1][:, None]
        nr_, nz_ = Nrm[:, 0][:, None], Nrm[:, 1][:, None]
        for pj in range(int(n_quad)):
            rp_, zp_ = qp[:, pj, 0][None, :], qp[:, pj, 1][None, :]
            nrp_, nzp_ = Nrm[:, 0][None, :], Nrm[:, 1][None, :]
            ap, bp, app, bpp, a, b = _coeffs(r_, z_, nr_, nz_, rp_, zp_, nrp_, nzp_)
            f1l, f1h = _facing_bounds(ap, bp)
            f2l, f2h = _facing_bounds(app, bpp)
            lo, hi = np.maximum(f1l, f2l), np.minimum(f1h, f2h)
            acc = np.zeros((m, m))
            for g in range(max_gaps):
                t0 = np.maximum(Gs[:, :, g], lo)
                t1 = np.minimum(Ge[:, :, g], hi)
                live = t1 > t0 + 1e-15
                if not live.any():
                    continue
                acc += np.where(
                    live,
                    _prim(ap, bp, app, bpp, a, b, np.where(live, t1, 0.0))
                    - _prim(ap, bp, app, bpp, a, b, np.where(live, t0, 0.0)),
                    0.0,
                )
            num += (2.0 / np.pi) * acc * aq[:, qi][:, None] * aq[:, pj][None, :]
    return num / aq.sum(1)[:, None]
