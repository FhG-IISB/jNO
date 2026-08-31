"""Enclosure radiation handle: ``domain.enclosure(tags)``.

An :class:`Enclosure` turns a set of boundary tags into the geometric ingredients an enclosure-radiation
boundary condition needs — built directly from the **FEM mesh boundary edges**, so the view-factor rows
align with global mesh nodes (and therefore FEM DOFs for a P1 field). The radiating surface is
discretised into **boundary elements** (mesh edges in 2D / meridional segments in the axisymmetric
``(r, z)`` plane); the element-to-element view-factor matrix ``F`` is *fully geometry-determined* —
occlusion (visibility) and orientation (cosine) decide every entry, only the ``i == i`` self-pair is
removed. Tags are merely element groups (for per-surface emissivity); they never block exchange, so a
concave surface keeps its self-view.

The radiosity physics (``q = (I-F)(I-diag(rho)F)^-1 diag(eps) sigma T^4``) is written by the user in
``jno.np`` on top of ``enclosure.view_factor``; this module only supplies the quality-gated ``F``, the
element/node bookkeeping, and the per-element measure used to scatter a flux back onto FEM nodes.

Reference: M. F. Modest, *Radiative Heat Transfer*, 3rd ed., Ch. 4-5 (view factors; the net-radiation /
radiosity method for diffuse-grey enclosures).
"""

from __future__ import annotations

from functools import partial
from typing import List, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..trace import Placeholder
from .mesh_utils import MeshUtils


def _boundary_edge_third_vertex(triangles: np.ndarray):
    """Map each boundary edge (a tuple of two global node indices, sorted) to the third vertex of its
    one adjacent triangle. Boundary edges occur in exactly one triangle; the third vertex fixes the
    outward (out-of-solid) normal direction."""
    third: dict[tuple[int, int], int] = {}
    seen: dict[tuple[int, int], int] = {}
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for i0, i1, opp in ((a, b, c), (b, c, a), (c, a, b)):
            key = (i0, i1) if i0 < i1 else (i1, i0)
            seen[key] = seen.get(key, 0) + 1
            third[key] = opp
    return third, seen


def _element_visibility(mids: np.ndarray, own_edge: np.ndarray, occ0: np.ndarray, occ1: np.ndarray):
    """Element-to-element visibility (0/1): element *i* sees *j* iff the segment between their midpoints
    crosses no occluder edge (excluding each element's own edge). Segment-edge intersection, vectorised
    over occluders per source (mirrors :meth:`MeshUtils.get_visibility_matrix_raytrace`)."""
    m = mids.shape[0]
    edge_dir = occ1 - occ0  # (E, 2)

    def cross2d(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    vm = np.ones((m, m), dtype=np.float64)
    idx = np.arange(m)
    for i in range(m):
        ab = mids - mids[i]  # (m, 2)
        denom = cross2d(ab[:, None, :], edge_dir[None, :, :])  # (m, E)
        parallel = np.abs(denom) < 1e-12
        denom_safe = np.where(parallel, 1.0, denom)
        diff = (occ0 - mids[i])[None, :, :]  # (1, E, 2)
        t_seg = cross2d(diff, edge_dir[None, :, :]) / denom_safe
        t_edge = cross2d(diff, ab[:, None, :]) / denom_safe
        eps = 1e-9
        crossing = (~parallel) & (t_seg > eps) & (t_seg < 1 - eps) & (t_edge > eps) & (t_edge < 1 - eps)
        crossing[:, own_edge[i]] = False  # source element's own edge never blocks
        crossing[idx, own_edge] = False  # each target element's own edge never blocks
        vm[i, :] = (~np.any(crossing, axis=1)).astype(np.float64)
    vm[idx, idx] = 0.0
    return vm


def _solid_polygon_visibility(domain, elem_tag, mids, normals, length):
    """Element-to-element visibility using the clean solid geometry: element *i* sees *j* iff the segment
    between their midpoints (nudged into the medium along the element normals) does not pass through any
    opaque solid region's interior. Uses ``domain._source_regions`` (the exact polygons), so it is immune
    to mesh-classification slivers. Exact for a genuinely 2-D (planar) enclosure. For an axisymmetric
    enclosure this straight-line test is only the ``phi = 0`` (same-meridian) slice of the true 3-D
    occlusion -- see :func:`_solid_polygon_visibility_3d`, which callers should use instead when
    ``axisymmetric=True``; a solid ring does NOT block every azimuth in general (the true 3-D chord
    between two rings bows closer to the axis than this flat projection as the azimuthal offset grows,
    so a same-meridian block/non-block verdict does not transfer to other azimuths)."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    from shapely.prepared import prep

    m = mids.shape[0]
    regions = getattr(domain, "_source_regions", {}) or {}
    geoms = [regions[s] for s in sorted(set(map(str, elem_tag))) if s in regions]
    if not geoms:
        return 1.0 - np.eye(m)
    union = unary_union(geoms)
    prepared = prep(union)
    eps = 0.5 * float(np.median(length))  # nudge endpoints off the solid boundary into the medium
    tol = 0.25 * eps
    P = mids + eps * np.asarray(normals)
    vm = np.eye(m) * 0.0
    for i in range(m):
        for j in range(i + 1, m):
            seg = LineString((P[i], P[j]))
            vis = 1.0
            if prepared.intersects(seg):  # cheap filter; only then measure the crossing length
                if float(getattr(seg.intersection(union), "length", 0.0)) > tol:
                    vis = 0.0
            vm[i, j] = vm[j, i] = vis
    return vm


def _min_solid_thickness(geoms, hi: float) -> float:
    """Thickness of the THINNEST opaque solid = 2 x its largest inscribed circle, by bisection on
    ``buffer(-d)``. This is the length scale the chord-vs-solid test must resolve, and it is set by the
    GEOMETRY (a 0.5 mm seed disc), not by the mesh -- which is exactly why neither the crossing-length
    tolerance nor the sampling stride may be scaled to the element size."""
    w = hi
    for g in geoms:
        lo, up = 0.0, float(min(g.bounds[2] - g.bounds[0], g.bounds[3] - g.bounds[1])) * 0.5 + 1e-12
        for _ in range(24):  # bisect the largest d with a non-empty erosion = the inscribed radius
            d = 0.5 * (lo + up)
            if g.buffer(-d).is_empty:
                up = d
            else:
                lo = d
        w = min(w, 2.0 * lo)
    return max(w, 1e-9)


class _SolidSDF:
    """Signed-distance field of the opaque-solid union on the ``(r, z)`` meridian (negative inside).

    Exists to make the chord-vs-solid test both CORRECT and affordable, via sphere tracing (below).

    Fixed-stride sampling of a chord cannot do that job: the stride has to be shorter than the thinnest
    solid or the chord steps straight over it, and in a real furnace that ratio is brutal (a 0.5 mm seed
    disc against ~500 mm chords needs ~3000 samples per chord, over ~650k pairs x n_phi azimuths). And a
    missed wall is not a benign inaccuracy -- it is occlusion that silently does not happen, so the ray
    passes THROUGH a solid, inflating F and pushing row sums past the physical bound of 1. Measured on the
    cg furnace, a 16-sample chord test wrongly called 12% of all *visible* pairs visible.

    A distance field fixes both at once: you may safely advance by the distance to the nearest solid,
    because nothing can be hit within it. Steps are therefore LARGE in open space and shrink automatically
    as a wall is approached -- and, crucially, a feature can never be skipped, whatever its size. That is
    a correctness guarantee that no fixed stride can give.

    This works because the solids are bodies of REVOLUTION: the 3-D distance from a point to a revolved
    solid equals the 2-D distance from ``(r, z)`` to its meridian polygon, so one 2-D field serves the
    full 3-D query.

    Reference: J. C. Hart, "Sphere tracing: a geometric method for the antialiased ray tracing of implicit
    surfaces", The Visual Computer 12 (1996) 527-545.
    """

    __hash__ = object.__hash__  # static jit arg: one SDF instance == one compiled march
    __eq__ = object.__eq__

    def __init__(self, union, bounds, cell: float, pad: float):
        from scipy.ndimage import distance_transform_edt
        from shapely.vectorized import contains as _vcontains

        # The grid must span every point a CHORD can visit, not merely the solid's own bounding box: a
        # query outside the grid is clamped to its edge, so a field sized to the solid alone reports a
        # tiny clearance everywhere outside it, and the march then crawls and never arrives. `bounds` is
        # therefore the union of the solid extent and the radiating elements' extent (a chord's meridian
        # track stays inside the latter: rho(t) <= max(r_i, r_j) and z(t) lies between z_i and z_j).
        r0 = min(union.bounds[0], bounds[0], 0.0) - pad  # include the axis: rho(t) can reach 0
        z0 = min(union.bounds[1], bounds[1]) - pad
        r1 = max(union.bounds[2], bounds[2]) + pad
        z1 = max(union.bounds[3], bounds[3]) + pad
        self.cell = float(cell)
        self.r0, self.z0 = r0, z0
        nr = int(np.ceil((r1 - r0) / cell)) + 1
        nz = int(np.ceil((z1 - z0) / cell)) + 1
        gr = r0 + (np.arange(nr) + 0.5) * cell
        gz = z0 + (np.arange(nz) + 0.5) * cell
        RR, ZZ = np.meshgrid(gr, gz, indexing="ij")
        occ = np.asarray(_vcontains(union, RR.ravel(), ZZ.ravel())).reshape(nr, nz)
        # signed: +distance to the solid outside it, -depth inside it
        d_out = distance_transform_edt(~occ) * cell
        d_in = distance_transform_edt(occ) * cell
        self.sdf = (d_out - d_in).astype(np.float64)
        self.nr, self.nz = nr, nz
        self.jax = jnp.asarray(self.sdf)  # eager: see the note on the annotation above

    #: The field as a device array. Built EAGERLY in ``__init__`` -- materialising it lazily inside
    #: the jitted march would cache a tracer and leak it into every later call.
    jax: "jnp.ndarray"

    def signed_jax(self, r, z):
        """:meth:`signed`, in JAX. Same clamping and same out-of-grid convention (a query can only
        fall outside the chord-reachable grid by round-off, and such points are far from any solid)."""
        fr = (r - self.r0) / self.cell
        fz = (z - self.z0) / self.cell
        ir = fr.astype(jnp.int32)
        iz = fz.astype(jnp.int32)
        ok = (ir >= 0) & (ir < self.nr) & (iz >= 0) & (iz < self.nz)
        ir = jnp.clip(ir, 0, self.nr - 1)
        iz = jnp.clip(iz, 0, self.nz - 1)
        return jnp.where(ok, self.jax[ir, iz], jnp.inf)

    def signed(self, r, z):
        """Signed distance at ``(r, z)``. The grid spans the whole chord-reachable region, so a query can
        only fall outside it by round-off; such points are far from any solid, hence a large clearance."""
        fr = (r - self.r0) / self.cell
        fz = (z - self.z0) / self.cell
        ir = fr.astype(np.int32)
        iz = fz.astype(np.int32)
        ok = (ir >= 0) & (ir < self.nr) & (iz >= 0) & (iz < self.nz)
        np.clip(ir, 0, self.nr - 1, out=ir)
        np.clip(iz, 0, self.nz - 1, out=iz)
        return np.where(ok, self.sdf[ir, iz], np.inf)


@partial(jax.jit, static_argnums=(0, 8))
def _march_jax(sdf, ri, zi, dx, dy, dz, Lsafe, depth, max_iter):
    """Sphere-trace every chord in lockstep. Returns ``(t, blocked, active)``; ``active`` marks the
    rays that never resolved -- the grazing minority the caller finishes with a dense scan.

    ``sdf`` is static (hashed by identity): it carries the grid metadata the indexing needs, and the
    field itself rides in as a device array via :attr:`_SolidSDF.jax`."""

    def probe(tt):
        x = ri + tt * dx
        y = tt * dy
        z = zi + tt * dz
        return sdf.signed_jax(jnp.sqrt(x * x + y * y), z) + depth

    min_step = 0.5 * sdf.cell / Lsafe  # never stall: always advance at least half a cell

    def cond(state):
        _, _, active, k = state
        return jnp.logical_and(k < max_iter, jnp.any(active))

    def body(state):
        t, blocked, active, k = state
        g = probe(t)
        hit = active & (g <= 0.0)
        blocked = blocked | hit
        active = active & (~hit)
        t = jnp.where(active, t + jnp.maximum(g / Lsafe, min_step), t)
        active = active & (t < 1.0)
        return t, blocked, active, k + 1

    t0 = jnp.zeros(ri.shape, dtype=jnp.float64)
    return jax.lax.while_loop(cond, body, (t0, jnp.zeros(ri.shape, bool), Lsafe > 1e-12, 0))[:3]


def _chord_blocked(sdf, ri, zi, rj, zj, cphi, sphi, depth, max_iter=160):
    r"""Sphere-trace the 3-D chord between two ring points; True where it penetrates a solid by ``depth``.

    The chord runs from ``(ri, phi=0, zi)`` to ``(rj, phi, zj)`` in 3-D; its meridian track is
    ``rho(t) = |(1-t) P_i + t P_j|_{xy}``, ``z(t) = (1-t) z_i + t z_j``. March ``t`` forward by the
    clearance to the (eroded) solid at the current point, converted to a step in ``t`` via the chord's
    true 3-D length. Because a step never exceeds the distance to the nearest solid, the march CANNOT
    jump over one -- unlike a fixed stride, which silently skips anything thinner than itself.

    Blocking is a **depth** criterion (``sdf < -depth``), not a crossing-length one: a chord that merely
    grazes along a solid's own face never gets deep, while one that truly passes through the thinnest wall
    reaches ``w/2`` at its mid-plane. Depth is also the right invariant for the endpoints, which sit ON
    radiating surfaces (``sdf ~ 0``) and so are never mistaken for a crossing -- no nudging required.

    **The grazing trap.** Sphere tracing converges only geometrically when a ray runs nearly PARALLEL to a
    surface: if the clearance shrinks by a fraction ``k`` of the distance travelled, the march needs
    ``~log(eps)/log(1-k)`` steps. On the cg furnace a chord skimming the 0.5 mm seed disc has ``k ~ 0.016``
    -- over 400 steps. Capping the iteration count and then treating the survivors as VISIBLE silently
    un-blocks every such ray, which is a hot-to-cold short circuit between surfaces that cannot see each
    other at all (it connected the crucible's inner cavity to the outer walls THROUGH 30 mm of solid seed).
    So rays that do not converge are NOT given a free pass: they fall back to a dense scan of whatever is
    left of the chord. That is expensive per ray, but only a small minority ever get there.

    Arrays broadcast over any leading shape; the endpoints may differ per element pair and per azimuth.
    """
    # broadcast endpoints AND azimuths together: the caller may vary phi per pair (graded rule) or hold
    # it fixed across a whole (i, j) block (uniform rule), so neither shape may be resolved on its own.
    ri, zi, rj, zj, cphi, sphi = np.broadcast_arrays(ri, zi, rj, zj, cphi, sphi)
    # 3-D endpoints: P_i on the phi = 0 meridian, P_j rotated by phi
    xj, yj = rj * cphi, rj * sphi
    dx, dy, dz = xj - ri, yj, zj - zi
    L = np.sqrt(dx * dx + dy * dy + dz * dz)
    Lsafe = np.maximum(L, 1e-300)

    def probe(tt, RI=None, ZI=None, DX=None, DY=None, DZ=None):
        """sdf clearance (eroded by `depth`) at chord parameter tt, on the full array or a subset."""
        RI = ri if RI is None else RI
        ZI = zi if ZI is None else ZI
        DX = dx if DX is None else DX
        DY = dy if DY is None else DY
        DZ = dz if DZ is None else DZ
        x = RI + tt * DX
        y = tt * DY
        z = ZI + tt * DZ
        return sdf.signed(np.sqrt(x * x + y * y), z) + depth

    # --- the march, in JAX: every (pair, azimuth) steps together, jitted, and it runs on whatever
    # device is active. Semantics are the numpy loop's, including the early exit once nothing is
    # active (a lax.while_loop, not a fixed fori_loop, so a cheap geometry still costs few steps).
    t, blocked, active = _march_jax(
        sdf,
        jnp.asarray(ri),
        jnp.asarray(zi),
        jnp.asarray(dx),
        jnp.asarray(dy),
        jnp.asarray(dz),
        jnp.asarray(Lsafe),
        float(depth),
        int(max_iter),
    )
    # np.array (not asarray): a device array converts to a READ-ONLY view, and the grazing fallback
    # below writes into `blocked` in place.
    t = np.array(t)
    blocked = np.array(blocked)
    active = np.array(active)

    # Rays still alive here never resolved -- they are grazing. Scan whatever is left of THEIR chords
    # densely rather than declaring them visible. Operate on the compressed subset only: these are a tiny
    # minority, and touching the whole array here would cost ~L/cell full-array probes (minutes, not ms).
    idx = np.flatnonzero(active.ravel())
    if idx.size:
        f = lambda a: np.broadcast_to(a, ri.shape).ravel()[idx]  # noqa: E731
        RI, ZI, DX, DY, DZ = f(ri), f(zi), f(dx), f(dy), f(dz)
        t0 = t.ravel()[idx]
        step = sdf.cell / np.maximum(f(L), 1e-300)  # never step past one cell -> cannot skip a feature
        n_dense = int(min(np.ceil(1.0 / max(float(step.min()), 1e-12)), 20000))
        hit_sub = np.zeros(idx.size, dtype=bool)
        for s in range(1, n_dense + 1):
            tt = np.minimum(t0 + s * step, 1.0)
            hit_sub |= probe(tt, RI, ZI, DX, DY, DZ) <= 0.0
            if (tt >= 1.0).all():
                break
        bl = blocked.ravel()
        bl[idx] |= hit_sub
        blocked = bl.reshape(ri.shape)
    return blocked


def _chord_test_setup(geoms, union, r, z):
    """Build the signed-distance field and the penetration depth that counts as a block.

    ``cell`` must resolve the THINNEST solid (``w/8``) -- never the mesh. Scaling the chord test to the
    element size (as the legacy straight-line test's ``tol`` does) is a trap: a solid thinner than the
    tolerance can then NEVER register as a block, at any sampling density. The cg furnace hits exactly
    that, with a 0.5 mm seed disc against ~1.2 mm elements. It is also capped against the domain span, so a
    geometry whose only solid is thick does not end up with an absurdly coarse field.

    ``depth`` is the penetration that counts as opaque, and it must be a small NUMERICAL tolerance -- a
    couple of raster cells -- not a fraction of the wall thickness. Testing ``sdf < -depth`` erodes every
    solid by ``depth``, so a depth tied to the thinnest wall would gouge huge chunks out of THICK ones and
    let grazing rays sail through them (it put the analytic concentric-cylinder ``F22`` at 0.506 vs 0.429).
    Two cells is the floor: the rasterised ``sdf`` can read up to ~1 cell negative for a point sitting
    exactly ON a surface, and every chord endpoint does exactly that -- a smaller depth would report every
    element as blocking itself.
    """
    w = _min_solid_thickness(geoms, hi=float(max(np.ptp(r), np.ptp(z))))
    span = float(np.hypot(np.ptp(r), np.ptp(z))) + 2.0 * float(np.max(r))
    cell = min(w / 8.0, span / 4000.0)
    bounds = (float(np.min(r)), float(np.min(z)), float(np.max(r)), float(np.max(z)))
    sdf = _SolidSDF(union, bounds, cell=cell, pad=8.0 * cell)
    return sdf, 2.0 * cell


def _solid_polygon_visibility_3d(domain, elem_tag, P, length, phi, n_seg: int = 0, occluders=None):
    r"""Point-to-point visibility **per azimuth** for the true 3-D ray between axisymmetric rings.

    ``P`` is an ``(M, 2)`` array of ``(r, z)`` points -- pass the kernel's own quadrature points
    (:meth:`MeshUtils.meridional_quad_points`) so the shadow boundary is resolved *within* an element,
    not just element midpoints (which makes a partially-shadowed element all-or-nothing).

    Point *i* sits at azimuth 0 (WLOG, by the enclosure's rotational symmetry); point *j* sits at
    azimuth ``phi``. The straight 3-D chord between them collapses EXACTLY onto a curve in the
    ``(r, z)`` meridian half-plane:

    .. math::
        \rho(t,\phi) = \sqrt{(1-t)^2 r_i^2 + 2t(1-t)r_i r_j\cos\phi + t^2 r_j^2},
        \qquad z(t) = (1-t)z_i + t z_j

    which reduces to the straight ``r_i -> r_j`` segment :func:`_solid_polygon_visibility` tests exactly
    at ``phi = 0``, and bows strictly closer to the axis than that flat projection for any ``phi != 0``
    (``rho(t,\phi)^2 - r_proj(t)^2 = 2t(1-t)r_i r_j(\cos\phi - 1) <= 0``). Occlusion at that azimuth is
    then: does this curve penetrate an opaque solid?

    Answered by SPHERE TRACING against a signed-distance field (:class:`_SolidSDF`,
    :func:`_chord_blocked`) rather than by sampling the chord at a fixed stride. A fixed stride has to be
    shorter than the thinnest solid or it steps clean over it -- and on this geometry that means thousands
    of samples per chord. Marching by the distance to the nearest solid instead takes LARGE steps through
    open space and small ones only near a wall, and can never skip a feature of any size.

    The endpoints are used **un-nudged**: the straight-line test nudges them off the surface along the
    element normals, which is harmless for a straight segment but here would pull both endpoints radially
    inward and make the curved chord bow spuriously close to the axis -- systematically over-reporting
    occlusion (caught against the analytic concentric-cylinder ``F22 = 1 - r1/r2``). No nudge is needed:
    the depth criterion already ignores endpoints lying ON their own radiating surface.

    Visibility is EVEN in ``phi`` (the chord depends on the azimuth only through ``cos phi`` and, in
    ``rho``, ``sin^2 phi``), so only the half-grid is computed and the rest mirrored.

    ``n_seg`` is accepted and ignored; it is a vestige of the fixed-stride implementation.

    ``occluders`` is the OPAQUE-SOLID model the chords are traced against:

    * a sequence of shapely meridian polygons -- exactly those solids block, nothing else;
    * ``()`` / an empty sequence -- an explicit "nothing occludes";
    * ``None`` -- fall back to deriving them from ``domain._source_regions`` keyed by the SET of
      ``elem_tag`` values.

    The fallback is the dangerous case, and it is why this argument exists. It makes a solid an
    occluder only if some radiating element happens to be *tagged* with it, so a solid that owns no
    radiating element -- or one whose elements were tagged under a different name -- is silently
    transparent. Measured on the cg furnace, tagging every insulation element ``WallS`` left ``WallU``,
    ``WallO`` and three quartz ports out of the occluder set, and **46% of the pairs this function
    called visible had chords passing through solid material** (worst case: 99% of the chord buried in
    insulation). There is no error and the resulting ``F`` looks perfectly plausible. Callers that know
    their opaque geometry should pass it rather than let it be inferred from tags.

    Returns an ``(M, M, len(phi))`` bool array (True = visible/unblocked at that azimuth); the self-pair
    is excluded at every azimuth.
    """
    from shapely.ops import unary_union

    P = np.asarray(P, dtype=np.float64)
    M = P.shape[0]
    phi = np.asarray(phi, dtype=np.float64)
    n_phi = phi.shape[0]
    if occluders is None:
        regions = getattr(domain, "_source_regions", {}) or {}
        geoms = [regions[s] for s in sorted(set(map(str, elem_tag))) if s in regions]
    else:
        geoms = list(occluders)
    vis = np.ones((M, M, n_phi), dtype=bool)
    if geoms:
        union = unary_union(geoms)
        r, z = P[:, 0], P[:, 1]
        sdf, depth = _chord_test_setup(geoms, union, r, z)
        # phi and 2*pi - phi give the same chord -> compute the half-grid, mirror the rest.
        half = [k for k in range(n_phi) if phi[k] <= np.pi + 1e-12]
        for k in half:
            c, s = float(np.cos(phi[k])), float(np.sin(phi[k]))
            v = np.empty((M, M), dtype=bool)
            for a in range(0, M, 128):  # row-chunked to bound the march's working set
                b = min(a + 128, M)
                v[a:b] = ~_chord_blocked(sdf, r[a:b, None], z[a:b, None], r[None, :], z[None, :], c, s, depth)
            vis[:, :, k] = v
            vis[:, :, (n_phi - k) % n_phi] = v  # phi -> 2*pi - phi
    # NOTE: the self-pair (i == i) is deliberately NOT forced to False. In an axisymmetric enclosure a
    # point at azimuth 0 and "itself" at azimuth phi are DIFFERENT physical points on the same ring, and
    # on a concave surface they genuinely exchange radiation -- that is real energy, not a degenerate
    # self-view (verified by Monte-Carlo: it is exactly the deficit that makes the row sums fall short).
    # It still costs nothing at phi = 0, where the chord degenerates to a point and the kernel's cosines
    # vanish identically.
    return vis


def _consistent_node_weights(areas: np.ndarray, endpoints, axisymmetric: bool) -> np.ndarray:
    r"""Per-element P1 load weights ``(m, 2)``: the two endpoint shares of ``\int_elem N_i \,d\Gamma``.

    **2D** — the measure is the edge length and ``N_i`` is symmetric about the midpoint, so each endpoint
    takes half: ``(L/2, L/2)``.

    **Axisymmetric** — the measure carries the ring Jacobian, ``d\Gamma = 2\pi r \,ds``, and ``r`` varies
    linearly along the element (``r(s) = r_0 + s\,\Delta``, ``\Delta = r_1 - r_0``). The halves are then
    *not* equal: the endpoint at the larger radius sweeps more area. With ``L`` the meridional length,

    .. math::
        \int_0^1 (1-s)\,r(s)\,2\pi L\,ds = \tfrac{\pi L}{3}(2r_0 + r_1), \qquad
        \int_0^1 s\,r(s)\,2\pi L\,ds     = \tfrac{\pi L}{3}(r_0 + 2r_1)

    which still sums to the ring area ``2\pi \bar{r} L`` but splits it ``(2r_0+r_1) : (r_0+2r_1)``. Using
    the 2D half-and-half split instead is exact only for an element at constant radius (a cylindrical
    wall, ``\Delta = 0``); for a radial element spanning ``0 \to R`` -- an end disc, which every closed
    axisymmetric cavity has -- it puts ``R/4`` on each node where the truth is ``R/6`` and ``R/3``, i.e.
    50% too much on the inner node. These weights integrate a linear test function EXACTLY (verified
    against ``\int 2\pi r \cdot r\,dr``), which the equal split does not.
    """
    areas = np.asarray(areas, dtype=np.float64)
    if not axisymmetric or endpoints is None:
        half = 0.5 * areas
        return np.stack([half, half], axis=1)
    e0, e1 = (np.asarray(e, dtype=np.float64) for e in endpoints)
    r0, r1 = e0[:, 0], e1[:, 0]
    length = np.linalg.norm(e1 - e0, axis=1)
    w0 = np.pi * length * (2.0 * r0 + r1) / 3.0
    w1 = np.pi * length * (r0 + 2.0 * r1) / 3.0
    return np.stack([w0, w1], axis=1)


class PendingElementExpr:
    """A per-element computation still waiting for the solution vector.

    Produced by :meth:`Enclosure.field` when it is handed a *symbolic* trial function rather than a
    concrete DOF vector, and consumed by :meth:`Enclosure.load`, which closes it into a
    ``jno.Coupling``. Arithmetic records the operation instead of performing it, so a nonlocal term is
    written as an expression in the ``jno.fem([...])`` list::

        gap.load(G @ gap.field(u) ** 4)              # an expression, sits in the term list
        lambda T: gap.load(G @ gap.field(T) ** 4)    # the same thing, written by hand

    This is not radiation-specific. Every nonlocal term in jNO has the same **gather -> operate ->
    scatter** shape -- integral and non-reflecting BCs, contact, peridynamics -- and ``field``/``load``
    are already the gather and the scatter, so deferring them covers all of them with one mechanism.
    The physics in between stays yours to write.

    .. important::
       The block this produces is **dense** in element space. It reads like a weak term and is not
       one: its Jacobian couples every element to every other, which is why ``fem.solve`` stays on
       matrix-free Newton-Krylov rather than assembling a sparse tangent.
    """

    __slots__ = ("_fn",)

    # NumPy checks this before trying to coerce an unknown right operand, so `ndarray @ pending`
    # reaches __rmatmul__ instead of being turned into an object array. JAX arrays already return
    # NotImplemented for unrecognized operands, which routes there too.
    __array_priority__ = 1000.0
    __array_ufunc__ = None

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, u):
        return self._fn(u)

    def _lift(self, other, op, swap=False):
        fn = self._fn
        if isinstance(other, PendingElementExpr):
            g = other._fn
            return PendingElementExpr(lambda u: op(g(u), fn(u)) if swap else op(fn(u), g(u)))
        return PendingElementExpr(lambda u: op(other, fn(u)) if swap else op(fn(u), other))

    def __add__(self, o):
        return self._lift(o, lambda a, b: a + b)

    def __radd__(self, o):
        return self._lift(o, lambda a, b: a + b, swap=True)

    def __sub__(self, o):
        return self._lift(o, lambda a, b: a - b)

    def __rsub__(self, o):
        return self._lift(o, lambda a, b: a - b, swap=True)

    def __mul__(self, o):
        return self._lift(o, lambda a, b: a * b)

    def __rmul__(self, o):
        return self._lift(o, lambda a, b: a * b, swap=True)

    def __truediv__(self, o):
        return self._lift(o, lambda a, b: a / b)

    def __rtruediv__(self, o):
        return self._lift(o, lambda a, b: a / b, swap=True)

    def __matmul__(self, o):
        return self._lift(o, lambda a, b: a @ b)

    def __rmatmul__(self, o):
        return self._lift(o, lambda a, b: a @ b, swap=True)

    def __pow__(self, o):
        return self._lift(o, lambda a, b: a**b)

    def __neg__(self):
        fn = self._fn
        return PendingElementExpr(lambda u: -fn(u))

    def apply(self, fn):
        """Record an arbitrary elementwise/array function, for anything the operators do not cover
        (``gap.field(u).apply(jnp.exp)``)."""
        inner = self._fn
        return PendingElementExpr(lambda u: fn(inner(u)))

    def __repr__(self):
        return "PendingElementExpr(<pending nonlocal term; pass to enclosure.load>)"


class Enclosure:
    """Geometric handle for an enclosure-radiation surface set (see module docstring).

    Attributes
    ----------
    view_factor : (m, m) jnp.ndarray
        Element-to-element diffuse view factor ``F`` (rows = receiving elements). Quality-gated on build.
    elements : (m, 2) np.ndarray
        Global mesh node-index pairs for each boundary element (endpoints are FEM nodes).
    element_tags : (m,) np.ndarray[object]
        The tag each element belongs to (for per-surface emissivity).
    areas : (m,) jnp.ndarray
        Per-element measure: edge length (2D) or ring area ``2*pi*r*length`` (axisymmetric).
    normals : (m, 2) jnp.ndarray
        Unit element normals pointing into the enclosure.
    nodes : (k,) np.ndarray
        Unique global node indices used by the enclosure (FEM DOFs for a scalar P1 field).
    """

    def __init__(self, domain, tags, F, elements, element_tags, areas, normals, midpoints, axisymmetric, endpoints=None):
        self.domain = domain
        self.tags = list(tags)
        self._F = jnp.asarray(F)
        self.elements = np.asarray(elements)
        self.element_tags = np.asarray(element_tags, dtype=object)
        self.areas = jnp.asarray(areas)
        self.normals = jnp.asarray(normals)
        self.midpoints = np.asarray(midpoints)
        self.axisymmetric = bool(axisymmetric)
        self._node_weights = jnp.asarray(_consistent_node_weights(np.asarray(areas), endpoints, self.axisymmetric))

    @property
    def size(self) -> int:
        return int(self._F.shape[0])

    @property
    def view_factor(self):
        return self._F

    @property
    def nodes(self) -> np.ndarray:
        return np.unique(self.elements)

    def tag_mask(self, tag: str) -> np.ndarray:
        """Boolean (m,) mask of elements belonging to ``tag`` — e.g. to build a per-element emissivity."""
        return self.element_tags == tag

    def emissivity(self, values) -> jnp.ndarray:
        """Per-element emissivity ``(m,)`` from a ``{tag: eps}`` mapping (or a scalar for all surfaces)."""
        if np.isscalar(values):
            return jnp.full(self.size, float(values))
        eps = np.zeros(self.size)
        for tag, val in dict(values).items():
            eps[self.tag_mask(tag)] = float(val)
        return jnp.asarray(eps)

    def field(self, u) -> jnp.ndarray:
        """Per-element temperature ``(m,)`` from the global solution ``u`` — the **nonlocal gather**.

        Radiosity is piecewise-constant per boundary element, so each element's temperature is the mean
        of its two endpoint (FEM node) values. Differentiable in ``u`` (used inside the Newton residual).

        Handed a **symbolic** trial function (``u, v = d.fem_symbols()``) instead of a concrete DOF
        vector, this returns a :class:`PendingElementExpr` instead: the gather is recorded, arithmetic
        on it is recorded, and :meth:`load` closes it into a ``jno.Coupling``. That is what lets a
        nonlocal term be written as an expression in the ``jno.fem([...])`` list rather than a lambda."""
        if isinstance(u, Placeholder):
            return PendingElementExpr(self.field)
        u = jnp.asarray(u).reshape(-1)
        return 0.5 * (u[self.elements[:, 0]] + u[self.elements[:, 1]])

    def load(self, q, *, size: Optional[int] = None) -> jnp.ndarray:
        """Consistent global surface load ``(n_dofs,)`` from a per-element flux ``q`` ``(m,)``.

        Scatters ``∫_Γ q v dΓ`` onto the FEM nodes for piecewise-constant ``q`` and P1 test functions.
        In 2D the two endpoints split the edge evenly; in **axisymmetric** mode the ring Jacobian
        ``2πr`` weights them by radius instead (see :func:`_consistent_node_weights`), which is what
        makes the load exact for a linear test function on a radial element. ``size`` defaults to the
        mesh node count (scalar P1 DOF layout).

        .. important::
           With ``axisymmetric=True`` this load is **per full revolution** (W, not W/m): the measure is
           ``2πr ds``. jNO does not weight the FEM forms for you, so the weak form this load is added to
           must carry the same ``2πr`` factor or the two sides differ by exactly that. See the
           *Axisymmetric* section of ``docs/fem.md``.

        Given a :class:`PendingElementExpr` (anything built from a symbolic ``field(u)``) this returns
        the ``jno.Coupling`` that closes it over the solution vector, so the whole nonlocal term is an
        expression that drops straight into the ``jno.fem([...])`` list."""
        if isinstance(q, PendingElementExpr):
            from .._fem import Coupling  # local: jno._fem imports the domain package

            tags = ",".join(map(str, self.tags)) if getattr(self, "tags", None) else "enclosure"
            return Coupling(lambda u: self.load(q(u), size=size), name=f"enclosure[{tags}]")
        q = jnp.asarray(q).reshape(-1)
        n = int(size) if size is not None else int(np.asarray(self.domain.mesh.points).shape[0])
        w = self._node_weights.astype(q.dtype)
        load = jnp.zeros(n, dtype=q.dtype)
        load = load.at[jnp.asarray(self.elements[:, 0])].add(q * w[:, 0])
        load = load.at[jnp.asarray(self.elements[:, 1])].add(q * w[:, 1])
        return load

    def quality(self):
        """Return ``(closure_error, reciprocity_error)`` for the assembled ``F`` (raw, un-normalized).

        ``closure_error = max|sum_j F_ij - 1|`` (should -> 0 for a closed enclosure) and
        ``reciprocity_error = max|A_i F_ij - A_j F_ji|`` (should -> 0 by reciprocity)."""
        F = np.asarray(self._F)
        A = np.asarray(self.areas)
        closure = float(np.abs(F.sum(axis=1) - 1.0).max())
        reciprocity = float(np.abs(A[:, None] * F - (A[:, None] * F).T).max())
        return closure, reciprocity

    def check(self, *, closure_atol: float = 5e-2, reciprocity_atol: float = 1e-7):
        """F-quality gate: raise if closure or reciprocity exceeds tolerance (mesh too coarse / normals
        mis-oriented / surfaces not enclosing). Returns ``self`` for chaining."""
        closure, reciprocity = self.quality()
        if reciprocity > reciprocity_atol:
            raise ValueError(
                f"Enclosure view factor fails reciprocity (max|A_i F_ij - A_j F_ji| = {reciprocity:.2e} "
                f"> {reciprocity_atol:.0e}); normals or geometry are inconsistent."
            )
        if closure > closure_atol:
            raise ValueError(
                f"Enclosure view factor fails closure (max|row_sum - 1| = {closure:.2e} > {closure_atol:.0e}); "
                "the surfaces may not form a closed enclosure or the mesh is too coarse."
            )
        return self

    def __repr__(self):
        c, r = self.quality()
        kind = "axisymmetric" if self.axisymmetric else "2D"
        return f"Enclosure({self.tags}, {kind}, {self.size} elements, closure={c:.1e}, reciprocity={r:.1e})"


def _pair_min_distance(e0: np.ndarray, e1: np.ndarray, ns: int = 5) -> np.ndarray:
    """Approximate min distance between every pair of meridional segments, by sampling ``ns`` points on
    each (error < h/(2*ns), and this only *selects* which pairs get refined -- refining a few extra is
    harmless, so an approximation is fine)."""
    m = e0.shape[0]
    t = ((np.arange(ns) + 0.5) / ns)[None, :, None]
    P = (e0[:, None, :] + t * (e1 - e0)[:, None, :]).reshape(m * ns, 2)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
    return D.reshape(m, ns, m, ns).min(axis=(1, 3))


def _graded_phi_rule(phi_c: np.ndarray, n_int: int, n_gl: int):
    r"""Per-pair azimuthal quadrature on ``[0, pi]``, GRADED toward ``phi = 0``.

    The ring kernel's azimuthal integrand has a peak at ``phi = 0`` of width ``phi_c ~ d/r`` (``d`` = the
    3-D separation of the two surface points, ``r`` = their radius), because
    ``R^2(phi) ~ d^2 + r^2 phi^2``. A uniform ``n_phi`` rule with ``dphi >> phi_c`` samples the peak's
    crest and multiplies it by a step far wider than the peak -- overshooting the integral by ``~dphi/phi_c``.
    That is the entire "corner artifact": two surfaces meeting in a sub-millimetre wedge have
    ``phi_c ~ 1e-3`` while ``dphi ~ 6e-2``, and the view factor comes out several times too large.

    So: composite Gauss-Legendre on intervals ``[0, phi_c]`` then GEOMETRICALLY spaced out to ``pi``,
    which resolves the peak at whatever width it happens to have. Integrating on ``[0, pi]`` (and doubling)
    is exact: the integrand is even in ``phi`` -- ``R^2``, both cosines and the occlusion chord all depend
    on ``phi`` only through ``cos phi`` and ``sin^2 phi``.

    Returns ``(phi, w)``, each ``(n_pairs, (n_int + 1) * n_gl)``.
    """
    gx, gw = np.polynomial.legendre.leggauss(n_gl)
    phi_c = np.clip(np.asarray(phi_c, dtype=np.float64), 1e-7, 0.5 * np.pi)[:, None]
    k = np.arange(n_int + 1)[None, :]
    # breakpoints: 0, phi_c, then geometric phi_c -> pi
    geo = phi_c * (np.pi / phi_c) ** (k / n_int)  # (n_pairs, n_int+1);  geo[:,0]=phi_c, geo[:,-1]=pi
    b = np.concatenate([np.zeros_like(phi_c), geo], axis=1)  # (n_pairs, n_int+2)
    lo, hi = b[:, :-1], b[:, 1:]  # (n_pairs, n_int+1)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    phi = (mid[:, :, None] + half[:, :, None] * gx[None, None, :]).reshape(phi_c.shape[0], -1)
    w = (half[:, :, None] * gw[None, None, :]).reshape(phi_c.shape[0], -1)
    return phi, w


@jax.jit
def _refine_block_jax(rq, zq, rp, zp, nq, np_, aq, ap, cphi, sphi, w_eff):
    """Exchange ``G_ij = A_i F_ij / (2 pi)`` for a block of near pairs, in one jitted expression.

    This is the arithmetic that dominates an enclosure build (measured: 42 s of a 65 s build on the cg
    furnace). It is pure batched array work -- a (pair, quad_i, quad_j, azimuth) kernel contracted down
    to one number per pair -- so it belongs in JAX, where it jits and runs on whatever device is
    active, rather than in numpy on the host.

    ``w_eff`` is the azimuthal weight already multiplied by the occlusion mask, so a blocked azimuth
    simply contributes nothing. Exchange is formed in its SYMMETRIC form, which is what makes
    reciprocity exact by construction when the caller splits it back into ``F``.
    """
    RQ, ZQ = rq[:, :, None, None], zq[:, :, None, None]
    RP, ZP = rp[:, None, :, None], zp[:, None, :, None]
    C, S = cphi[:, None, None, :], sphi[:, None, None, :]
    dx = RP * C - RQ
    dy = RP * S
    dz = ZP - ZQ
    R2 = dx * dx + dy * dy + dz * dz
    R = jnp.sqrt(R2 + 1e-300)
    cos_q = jnp.maximum(0.0, (nq[:, 0][:, None, None, None] * dx + nq[:, 1][:, None, None, None] * dz) / R)
    cos_p = jnp.maximum(
        0.0, -(np_[:, 0][:, None, None, None] * (dx * C + dy * S) + np_[:, 1][:, None, None, None] * dz) / R
    )
    K = cos_q * cos_p / (jnp.pi * R2 + 1e-300)  # (n, Q, Q, NP)
    # azimuthal integral over [0, 2pi) = 2 x integral over [0, pi]  (the integrand is even in phi)
    Iqp = 2.0 * jnp.einsum("nqpf,nf->nqp", K, w_eff)
    return jnp.einsum("nq,nqp,np->n", aq, Iqp, ap)


def _refine_near_pairs(
    F,
    e0,
    e1,
    normals,
    domain,
    elem_tag,
    n_phi,
    *,
    n_sub=4,
    n_gl=4,
    n_int=14,
    n_gl_phi=6,
    k_azim=3.0,
    k_merid=3.0,
    n_seg=256,
    chunk=96,
    log=None,
    occluders=None,
):
    """Recompute NEAR element pairs (and the diagonal) with adaptive quadrature; return the corrected F.

    The vectorised kernel uses a uniform ``n_phi`` azimuthal rule and ``n_quad`` meridional Gauss points.
    Both fail for near-touching pairs (see :func:`_graded_phi_rule`), which is what forces the ``r_min``
    near-field fudge and leaves raw row sums well above the physical bound of 1. Here every pair that the
    base rule cannot resolve --

        azimuthally:  2*pi/n_phi  >  (d/r) / k_azim        (the peak is narrower than one step)
        meridionally: max(h_i,h_j) > d / k_merid           (the elements are big compared to their gap)

    -- is recomputed with a graded azimuthal rule and a composite meridional rule, with NO ``r_min``
    softening. The self-pair (i == i) is always refined: a ring's own azimuthal self-view is the most
    singular pair there is (d -> 0).

    Exchange is computed in its symmetric form ``G_ij = A_i F_ij`` and split back, so reciprocity is exact
    by construction. Occlusion is evaluated per pair per refined azimuth on the element midpoints, by the
    same 3-D chord test as :func:`_solid_polygon_visibility_3d`.

    ``occluders`` is the OPAQUE-SOLID model to test the refined chords against. It must be supplied
    whenever anything occludes:

    * ``(geoms, union)`` -- shapely meridian polygons; refined chords are sphere-traced against them
      (the interface-mode model).
    * ``()`` / an empty sequence -- an explicit "nothing occludes" (a closed convex cavity).
    * ``None`` -- fall back to deriving the polygons from ``domain._source_regions`` keyed by
      ``elem_tag``.

    The fallback is the dangerous case and the reason this argument exists: in **boundary** mode the tags
    are surface names, not region names, so the lookup comes back empty and every refined near pair is
    silently recomputed as fully visible. On concentric cylinders that turns the outer wall's occlusion-
    limited self-view ``F22 = 1 - r1/r2`` into ``0.64`` at ``r1/r2 = 0.8`` (true value ``0.2``) and pushes
    row sums to ``1.44`` -- worse than the corner overshoot the refinement exists to fix. Callers that
    know their occluder model must therefore pass it rather than let it be guessed.
    """
    from shapely.ops import unary_union

    F = np.array(F, dtype=np.float64, copy=True)
    m = e0.shape[0]
    mids = 0.5 * (e0 + e1)
    length = np.linalg.norm(e1 - e0, axis=1)

    # --- which pairs can the base rule not resolve? ---
    D = _pair_min_distance(e0, e1)
    r_ref = np.maximum(np.maximum(mids[:, 0][:, None], mids[:, 0][None, :]), 1e-9)
    h = np.maximum(length[:, None], length[None, :])
    dphi = 2.0 * np.pi / n_phi
    need = (D < k_azim * dphi * r_ref) | (D < k_merid * h)
    np.fill_diagonal(need, True)  # the ring self-view: the ultimate near pair
    iu, ju = np.where(np.triu(need))
    if log:
        log(
            f"near-field: refining {len(iu)} of {m * (m + 1) // 2} unordered pairs "
            f"({100 * len(iu) / (m * (m + 1) / 2):.1f}%)"
        )

    if occluders is None:  # legacy: guess the solids from the element tags (see the docstring's warning)
        regions = getattr(domain, "_source_regions", {}) or {}
        geoms = [regions[s] for s in sorted(set(map(str, elem_tag))) if s in regions]
        union = unary_union(geoms) if geoms else None
    else:
        geoms, union = occluders if occluders else ([], None)
    # Occlusion by sphere tracing against a signed-distance field (see _chord_test_setup / _chord_blocked):
    # scaled to the thinnest SOLID, not to the mesh, and unable to step over a feature of any size.
    sdf = depth = None
    if union is not None:
        sdf, depth = _chord_test_setup(geoms, union, mids[:, 0], mids[:, 1])

    # --- composite meridional rule on each element: n_sub sub-intervals x n_gl Gauss points ---
    gx, gw = np.polynomial.legendre.leggauss(n_gl)
    sub = (np.arange(n_sub) / n_sub)[:, None]
    s_loc = (sub + (gx[None, :] + 1.0) * 0.5 / n_sub).reshape(-1)  # (Q,) in [0,1]
    w_loc = np.tile(gw * 0.5 / n_sub, n_sub)  # (Q,) sums to 1
    h_sub = length / (n_sub * n_gl)  # finest resolved meridional scale

    for c0 in range(0, len(iu), chunk):
        ii, jj = iu[c0 : c0 + chunk], ju[c0 : c0 + chunk]
        n = ii.size

        # quadrature points on each element of the pair
        def qpts(idx):
            p = e0[idx][:, None, :] + s_loc[None, :, None] * (e1 - e0)[idx][:, None, :]  # (n, Q, 2)
            w = w_loc[None, :] * length[idx][:, None]  # (n, Q) meridional ds
            return p[:, :, 0], p[:, :, 1], w

        rq, zq, wq = qpts(ii)
        rp, zp, wp = qpts(jj)
        nq, np_ = normals[ii], normals[jj]  # (n, 2) each
        aq, ap = rq * wq, rp * wp  # ring-area weights (2*pi cancels)

        # graded azimuthal rule sized to THIS pair's peak width phi_c = d/r
        d_eff = np.maximum(D[ii, jj], np.maximum(h_sub[ii], h_sub[jj]))
        phi, wphi = _graded_phi_rule(d_eff / r_ref[ii, jj], n_int, n_gl_phi)  # (n, NP)
        NP = phi.shape[1]
        cphi, sphi = np.cos(phi), np.sin(phi)

        # occlusion per (pair, azimuth), on the element midpoints -- sphere-traced, same test as the base
        vis = np.ones((n, NP), dtype=np.float64)
        if sdf is not None:
            vis = (
                ~_chord_blocked(
                    sdf,
                    mids[ii, 0][:, None],
                    mids[ii, 1][:, None],
                    mids[jj, 0][:, None],
                    mids[jj, 1][:, None],
                    cphi,
                    sphi,
                    depth,
                )
            ).astype(np.float64)

        # ring kernel at every (quad_i, quad_j, phi) -> the symmetric exchange G, in JAX
        G = _refine_block_jax(
            jnp.asarray(rq),
            jnp.asarray(zq),
            jnp.asarray(rp),
            jnp.asarray(zp),
            jnp.asarray(nq),
            jnp.asarray(np_),
            jnp.asarray(aq),
            jnp.asarray(ap),
            jnp.asarray(cphi),
            jnp.asarray(sphi),
            jnp.asarray(wphi * vis),
        )
        G = np.asarray(G)

        Aq, Ap = aq.sum(axis=1), ap.sum(axis=1)  # = A_i/(2*pi), A_j/(2*pi)
        F[ii, jj] = G / Aq
        F[jj, ii] = G / Ap  # reciprocity exact by construction
    return F


def _classify_triangles(domain, triangles: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Region name (from ``domain._source_regions``) containing each triangle's centroid, else ``None``.

    Used by the interface (``medium_tags``) path to find solid|medium radiating faces. The geometry
    regions tile the domain, so on a connected mesh every centroid lands in exactly one region."""
    regions = getattr(domain, "_source_regions", {}) or {}
    cent = pts[triangles].mean(axis=1)
    region_of = np.full(triangles.shape[0], None, dtype=object)
    if not regions:
        # A ``Shape.regions`` domain carries its regions as Shapes, not shapely geometries. They answer
        # the same question through ``Shape.contains`` (analytic CSG membership, already vectorised
        # over points), and dict order is declaration order == region priority, matching the first-hit
        # rule below. Without this an enclosure cannot be built on a Shape-built domain at all: every
        # centroid stays unclassified, so no solid|medium interface edge is ever found.
        for name, sub in (getattr(domain, "_shape_regions", {}) or {}).items():
            try:
                hit = np.asarray(sub.contains(cent[:, : sub.dim]), dtype=bool)
            except NotImplementedError:
                continue  # a swept/revolved region has no closed-form membership
            region_of[hit & (region_of == None)] = name  # noqa: E711
        return region_of
    try:
        from shapely.vectorized import contains as _vcontains

        for name, geom in regions.items():
            hit = np.asarray(_vcontains(geom, cent[:, 0], cent[:, 1]))
            region_of[hit & (region_of == None)] = name  # noqa: E711
    except Exception:
        from shapely.geometry import Point

        for i in range(cent.shape[0]):
            for name, geom in regions.items():
                if geom.contains(Point(cent[i])):
                    region_of[i] = name
                    break
    return region_of


def _enforce_closure(F, areas, n_iter: int = 200):
    """Correct a raw view-factor matrix to satisfy closure (``sum_j F_ij = 1``) and reciprocity
    (``A_i F_ij = A_j F_ji``) simultaneously — the standard consistency fix for view factors from an
    approximate / discretised kernel (where near-field rings are over- or under-resolved).

    Symmetric matrix scaling (Sinkhorn) on the exchange matrix ``G = A_i F_ij``: repeatedly rescale ``G``
    by ``sqrt(A_i / sum_j G_ij)`` on both sides (keeping ``G`` symmetric) until its row sums approach the
    areas; then ``F = G / A_i`` has unit row sums and exact reciprocity. Done once at build time (NumPy).

    Reference: A. M. van Leersum, "A method for determining a consistent set of radiation view factors
    from a set generated by a non-exact method", Int. Comm. Heat Mass Transfer 16 (1989) 83-94.
    """
    F = np.asarray(F, dtype=np.float64)
    A = np.asarray(areas, dtype=np.float64)
    G = A[:, None] * F
    G = 0.5 * (G + G.T)  # symmetrize -> exact reciprocity
    for _ in range(int(n_iter)):
        s = G.sum(axis=1)
        s = np.where(s < 1e-30, 1.0, s)
        sc = np.sqrt(A / s)
        G = G * sc[:, None] * sc[None, :]
    return jnp.asarray(G / A[:, None])


def build_enclosure(
    domain,
    tags: Sequence[str],
    *,
    axisymmetric: bool = False,
    # Raised from 3 when the axisymmetric azimuthal integral became exact. The sampled path leaned on
    # the near-field refinement to repair its near pairs; the closed form has no refinement stage, so
    # the MERIDIONAL rule is now the only discretisation and must carry that accuracy alone. Measured
    # on a closed cylindrical cavity: max row sum 1.029 at n_quad=3 against 1.009 at n_quad=6. The
    # cost is small because runtime is dominated by the quadrature-INDEPENDENT interval algebra --
    # 7.6 s -> 9.0 s on the reference furnace for a ~3x better closure.
    n_quad: int = 6,
    # Accepted and ignored on the axisymmetric path: the azimuth is integrated in closed form, so
    # there is no azimuthal grid. Still used by the 2-D kernel.
    n_phi: int = 16,
    opaque_tags: Optional[Sequence[str]] = None,
    medium_tags: Optional[Sequence[str]] = None,
    enforce_closure: bool = False,
    closure_iters: int = 200,
    occlude: bool = True,
    inward: bool = False,
    r_min: Optional[float] = None,
    near_field: bool = True,
) -> Enclosure:
    """Assemble an :class:`Enclosure` from radiating ``tags`` on a meshed 2D / axisymmetric domain.

    Two surface-discretisation modes:

    * **Boundary mode** (``medium_tags is None``, default): radiating elements are domain **boundary
      edges** (one adjacent triangle) whose endpoints lie on a ``tags`` surface. By default the normals
      point *out of the mesh* — the radiating surface faces an *un-meshed* gap (e.g. a vacuum between
      solid parts). Set ``inward=True`` when the radiating surfaces are instead the **outer walls of a
      meshed cavity** and radiation crosses the *meshed* interior (an oven/furnace filled with a
      transparent fluid): the normals then point *into the mesh* so the facing walls see one another.
    * **Interface mode** (``medium_tags`` given): the radiating gap is itself **meshed** (a participating
      but transparent medium such as a furnace gas/air). Radiating elements are the internal
      **solid|medium interface edges** between a ``tags`` region (a solid, by geometry-part name) and a
      ``medium_tags`` region; normals point out of the solid into the medium. This is the common furnace
      case where every region is meshed.

    ``near_field`` (axisymmetric only, default on) recomputes the element pairs that the uniform ``n_phi``
    azimuthal rule cannot resolve — near-touching surfaces, e.g. two parts meeting in a narrow wedge, and
    every element's own ring self-view — with a graded azimuthal quadrature sized to each pair's peak (see
    :func:`_refine_near_pairs`). WITHOUT it, such pairs overshoot by roughly ``dphi / (d/r)``: a
    sub-millimetre gap at ~100 mm radius is off by several times, driving row sums far above the physical
    bound of 1. It costs a one-off build-time pass over the near pairs (typically a few % of all pairs).

    ``r_min`` (axisymmetric only) softens the ring kernel's near-field ``1/R^2`` singularity. It defaults
    to **0 when** ``near_field`` **is on** (the refinement resolves the near field properly, so no fudge is
    needed) and to half the median element length otherwise — that legacy default is itself a ~12% error on
    the analytic concentric-cylinder factors, because it is sized to mask the near-field failure rather
    than to fix it. Pass an explicit value to override either way.
    """
    if isinstance(tags, str):
        tags = [tags]
    tags = [str(t) for t in tags]
    if not tags:
        raise ValueError("domain.enclosure requires at least one boundary tag.")
    if int(getattr(domain, "dimension", 0)) != 2:
        raise NotImplementedError(
            "domain.enclosure is implemented for 2D and axisymmetric (r, z) meshes; 3D enclosure radiation is future work."
        )

    mesh = domain.mesh
    pts = np.asarray(mesh.points)[:, :2]
    triangles = np.asarray(mesh.cells_dict["triangle"])
    boundary_edges = np.asarray(MeshUtils.extract_boundary_edges(jnp.asarray(triangles), pts.shape[0]))

    # Per-tag node masks (global node indices on each radiating surface). Prefer the spatial predicate
    # registered by ``domain.tag`` (it takes coordinate arrays); fall back to a geometry-part node set.
    predicates = getattr(domain, "_tag_predicates", {}) or {}
    tag_indices = getattr(domain, "tag_indices", {}) or {}

    def _tag_mask(tag: str) -> np.ndarray:
        if tag in predicates:
            cols = [pts[:, i] for i in range(pts.shape[1])]
            return np.asarray(predicates[tag](*cols)).astype(bool)
        if tag in tag_indices:
            mask = np.zeros(pts.shape[0], dtype=bool)
            mask[np.asarray(tag_indices[tag], dtype=np.int64)] = True
            return mask
        raise ValueError(
            f"domain.enclosure: tag '{tag}' has neither a spatial predicate (domain.tag) nor mesh node "
            "indices; define it with domain.tag(name, predicate) or as a geometry part."
        )

    elem_nodes_list: List[tuple] = []
    elem_tag: List[str] = []
    if medium_tags is None:
        # Boundary mode: radiating elements are domain boundary edges whose endpoints lie on a tag.
        third, _ = _boundary_edge_third_vertex(triangles)
        tag_masks = {tag: _tag_mask(tag) for tag in tags}
        for a, b in boundary_edges:
            a, b = int(a), int(b)
            for tag in tags:
                mk = tag_masks[tag]
                if mk[a] and mk[b]:
                    elem_nodes_list.append((a, b))
                    elem_tag.append(tag)
                    break
        if not elem_nodes_list:
            raise ValueError(f"domain.enclosure: no boundary elements found on tags {tags}.")
    else:
        # Interface mode: radiating elements are internal solid|medium interface edges. ``tags`` are
        # solid geometry-part names; ``medium_tags`` are the transparent (meshed) media. The normal of
        # each element points out of the solid into the medium (its solid-side third vertex sets it).
        medium_set = {str(m) for m in medium_tags}
        tag_set = set(tags)
        tri_region = _classify_triangles(domain, triangles, pts)
        edge_adj: dict = {}
        for ti in range(triangles.shape[0]):
            a, b, c = int(triangles[ti, 0]), int(triangles[ti, 1]), int(triangles[ti, 2])
            rg = tri_region[ti]
            for i0, i1, opp in ((a, b, c), (b, c, a), (c, a, b)):
                key = (i0, i1) if i0 < i1 else (i1, i0)
                edge_adj.setdefault(key, []).append((rg, opp))
        third = {}
        for key, adj in edge_adj.items():
            if len(adj) != 2:
                continue
            (r0, t0), (r1, t1) = adj
            if r0 in tag_set and r1 in medium_set:
                elem_nodes_list.append(key)
                elem_tag.append(r0)
                third[key] = t0
            elif r1 in tag_set and r0 in medium_set:
                elem_nodes_list.append(key)
                elem_tag.append(r1)
                third[key] = t1
        if not elem_nodes_list:
            raise ValueError(
                f"domain.enclosure: no solid|medium interface elements found between tags {tags} and "
                f"medium_tags {list(medium_tags)}."
            )
    elem_nodes = np.asarray(elem_nodes_list, dtype=np.int64)  # (m, 2)
    elem_tag_arr = np.asarray(elem_tag, dtype=object)

    e0 = pts[elem_nodes[:, 0]]
    e1 = pts[elem_nodes[:, 1]]
    mids = 0.5 * (e0 + e1)
    edge_vec = e1 - e0
    length = np.linalg.norm(edge_vec, axis=1)

    # Into-enclosure normals: perpendicular to the edge, pointing away from the adjacent triangle's
    # third vertex (i.e. out of the solid -> into the gap).
    normals = np.zeros_like(mids)
    for k in range(elem_nodes.shape[0]):
        a, b = int(elem_nodes[k, 0]), int(elem_nodes[k, 1])
        key = (a, b) if a < b else (b, a)
        c = third.get(key)
        t = edge_vec[k]
        n = np.array([-t[1], t[0]])
        n = n / (np.linalg.norm(n) + 1e-30)
        if c is not None and np.dot(mids[k] - pts[c], n) < 0:
            n = -n  # ensure it points away from the solid interior
        normals[k] = n
    if inward and medium_tags is None:
        normals = -normals  # enclosure is the meshed cavity (e.g. an oven): normals point into the mesh

    # Occluders for visibility: all participating element edges (+ optional opaque-tag boundary edges).
    occ0_list, occ1_list = [e0], [e1]
    if opaque_tags:
        for otag in opaque_tags:
            om = _tag_mask(otag)
            sel = np.array([om[int(a)] and om[int(b)] for a, b in boundary_edges], dtype=bool)
            occ0_list.append(pts[boundary_edges[sel, 0]])
            occ1_list.append(pts[boundary_edges[sel, 1]])
    occ0 = np.concatenate(occ0_list, axis=0)
    occ1 = np.concatenate(occ1_list, axis=0)
    own_edge = np.arange(elem_nodes.shape[0])  # each element's own edge is the first m occluders
    # In interface mode the opaque solids are the geometry regions the radiating elements bound, plus
    # every OTHER region that is not a transparent medium: a solid that owns no radiating element still
    # blocks rays. Resolving this once, here, is what keeps the visibility test and the near-field
    # refinement on the same occluder model (and off the tag-derived fallback -- see
    # _solid_polygon_visibility_3d's docstring for what that silently costs).
    solid_geoms = None
    if medium_tags is not None:
        _regions = getattr(domain, "_source_regions", {}) or {}
        _transparent = {str(t) for t in medium_tags}
        solid_geoms = [g for name, g in _regions.items() if str(name) not in _transparent]
        if not solid_geoms and getattr(domain, "_shape_regions", None):
            # A Shape.regions domain has no shapely regions, so the line above yields NOTHING and the
            # enclosure would be built with an empty occluder model -- every pair mutually visible,
            # through solid metal and insulation alike. It fails silently: closure and reciprocity stay
            # perfect (they are enforced), so the F that comes out looks entirely plausible. Measured on
            # the cg furnace, one Quartz.1->WallO pair read 0.024 unoccluded against 0.363 occluded, and
            # the coupled solve landed 620 K cold.
            #
            # The mesh already carries the answer: `tri_region` labels every cell, so each opaque
            # region's meridian outline is the union of its own cells. Exact for straight-edged regions
            # (the cells tile them), and it needs no shapely geometry from the caller.
            from shapely.geometry import Polygon as _SPoly
            from shapely.ops import unary_union as _uu

            for _name in sorted({str(x) for x in tri_region if x is not None} - _transparent):
                _sel = np.where(tri_region == _name)[0]
                if not len(_sel):
                    continue
                _g = _uu([_SPoly(pts[triangles[t]][:, :2]) for t in _sel]).buffer(0)
                if not _g.is_empty:
                    solid_geoms.append(_g)

    # The AXISYMMETRIC path resolves visibility analytically inside the kernel, so none of the
    # sampled machinery below is built for it. That is where the speedup lives: the expensive step was
    # never the kernel arithmetic but `_solid_polygon_visibility_3d`, which rasterises a signed-distance
    # field whose cell size is set by the thinnest solid in the scene and then sphere-traces every pair
    # at every azimuth against it.
    if axisymmetric:
        vm = None
    elif not occlude:
        vm = 1.0 - np.eye(elem_nodes.shape[0])  # diagnostic: no occlusion (all mutually visible)
    elif medium_tags is not None:
        # Interface mode: occlude with the CLEAN solid polygons (a ray is blocked iff it passes through
        # a solid interior), immune to the mesh-sliver artefacts that the element-edge occluder suffers.
        if axisymmetric:  # unreachable: handled analytically above
            # Per-azimuth occlusion (see _solid_polygon_visibility_3d): a same-meridian (phi=0) verdict
            # does not transfer to other azimuths for a general solid of revolution, so this checks the
            # true 3-D chord at every azimuth the exchange kernel integrates over.
            #
            # Evaluated at element MIDPOINTS, not at the kernel's quadrature points. Resolving visibility
            # per quadrature point is ~10x more expensive (n_quad^2 more chord tests) and, measured
            # against the analytic concentric-cylinder factors, changes nothing (F12/F21/F22 and the row
            # sums agree to 3 decimals) -- the shadow boundary is smooth in azimuth, which is where it is
            # already resolved to n_phi. The kernel still ACCEPTS quadrature-resolved visibility if a
            # caller wants it (see get_view_factor_axisymmetric_element).
            phi_occ = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
            vm = _solid_polygon_visibility_3d(domain, elem_tag_arr, mids, length, phi_occ, occluders=solid_geoms)
        else:
            vm = _solid_polygon_visibility(domain, elem_tag_arr, mids, normals, length)
    else:
        vm = _element_visibility(mids, own_edge, occ0, occ1)

    if axisymmetric:
        # The azimuthal integral is evaluated in CLOSED FORM (view_factor_closed), which subsumes three
        # stages of the old path at once: the sampled azimuthal rule, the sphere-traced visibility, and
        # the near-field refinement that existed to repair both. Consequences worth stating:
        #
        #   * `n_phi` and `r_min` no longer mean anything. There is no azimuthal grid to refine and no
        #     1/R^2 singularity to floor, so the ~12% r_min bias and the O(1/n_phi) closure error are
        #     gone rather than tuned. Both are accepted and ignored (see the signature note).
        #   * Occlusion boundaries are algebraic, so a shadow edge lands exactly instead of on the
        #     nearest azimuthal bin (measured 113.265974 deg against an analytic 113.265974, where the
        #     32-bin rule gave 115.00).
        #   * Cost drops sharply because there is no signed-distance raster to build, and that raster's
        #     cell size was set by the THINNEST solid in the scene -- which is why the old path made a
        #     189-element cavity cost twice a 477-element chamber. Measured on the reference furnace:
        #     106.3 s -> 9.0 s, with closure simultaneously better (2.8e-02 vs 9.2e-02 on the cavity,
        #     3.4e-02 vs 4.2e-02 on the chamber, both before any enforce_closure).
        #
        # `vm` is unused here: visibility is resolved inside the kernel from the occluder geometry.
        from .view_factor_closed import segments_from_polygons, view_factor_axisymmetric_closed

        if medium_tags is not None:
            # interface mode: the clean solid polygons, exactly the set the old visibility test used
            _sides = segments_from_polygons(solid_geoms or [])
        elif occlude:
            # boundary mode: the element edges themselves (plus any opaque_tags edges) are the blockers
            _sides = np.c_[occ0[:, 0], occ0[:, 1], occ1[:, 0], occ1[:, 1]]
        else:
            _sides = np.zeros((0, 4))  # the caller asserts nothing blocks any ray
        F = jnp.asarray(view_factor_axisymmetric_closed(e0, e1, normals, occluders=_sides, n_quad=n_quad))
        areas = 2.0 * np.pi * mids[:, 0] * length  # ring areas
    else:
        F = MeshUtils.get_view_factor_2d_element(
            jnp.asarray(e0), jnp.asarray(e1), jnp.asarray(normals), jnp.asarray(vm), n_quad=n_quad
        )
        areas = length

    if enforce_closure:
        F = _enforce_closure(F, areas, n_iter=closure_iters)

    return Enclosure(domain, tags, F, elem_nodes, elem_tag_arr, areas, normals, mids, axisymmetric, endpoints=(e0, e1))
