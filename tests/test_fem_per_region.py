"""Per-region (sub-domain) volume integration in ``jno.fem``.

A weak term written on a *sub-region's* coordinates integrates over that sub-region's cells only,
instead of the whole domain. The region is carried by the coordinate tag (exactly like a boundary
term), and a cell belongs to a region iff its **centroid** does -- classified once at assembly build
time against a ``domain.tag`` predicate (or a geometry part). Internally the term is multiplied by a
``RegionMask`` leaf that the assembly kernel resolves from a constant per-cell ``volume_var``; because the
mask is just a scalar coefficient on the integrand, it composes with every coefficient kind
(constant / ``jno.fn`` / ``.freeze()`` / trainable) and every solve form.

These tests lock the headline behaviours:
  * the mask restricts integration **exactly** to the region's cells (a region load integrates to the
    region's cell area);
  * a region operator term **converges** to the same problem written with a ``jno.fn`` indicator
    coefficient (centroid vs quad-point differ only on interface cells -> O(h));
  * it works in **nonlinear**, **multifield**, **transient**, and **3D** forms;
  * a **scalar parameter trained on a sub-region** is recovered through ``crux``;
  * the one not-yet-wired path (second-order-in-time) **fails loud**, never silently whole-domain.
"""

from __future__ import annotations

import numpy as np
import pytest

import jno

pytest.importorskip("shapely", reason="shapely required for PolygonDomain")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from shapely.geometry import box  # noqa: E402

from jno.utils.solver.newton_krylov import newton_krylov  # noqa: E402

_DISK = lambda x, y: (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.2**2  # noqa: E731  (an interior sub-region)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _disk_setup(mesh_size=0.08):
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=mesh_size)
    d.tag("disk", _DISK)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xd, yd, _ = d.variable("disk", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    return (
        d,
        u,
        phi,
        (xi, yi),
        (xb, yb),
        (u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)),
        (u.bind(x=xd, y=yd), phi.bind(x=xd, y=yd)),
    )


def _solve(fem):
    return np.linalg.solve(np.asarray(fem.A), np.asarray(fem.b).reshape(-1))


def _indicator(xi, yi):
    return jno.fn(lambda x, y: jnp.where(_DISK(x, y), 1.0, 0.0), [xi, yi])


# ==========================================================================
# the mask restricts integration EXACTLY to the region's cells
# ==========================================================================
def test_region_load_integrates_over_region_cells_only():
    """A region-restricted load ``-1*v`` has ``sum(b) = area of the disk's cells (~ pi r^2)``; a
    whole-domain load over the interior sub-region equals the full area. Exact (not O(h)) because it
    is literally the integral of the per-cell indicator."""
    d, u, phi, (xi, yi), (xb, yb), (ui, vi), (ud, vd) = _disk_setup(mesh_size=0.05)
    disk = jno.fem([ui.x * vi.x + ui.y * vi.y, -1.0 * vd, u(xb, yb) - 0.0])
    area = float(np.asarray(disk.b).sum())
    assert abs(area - np.pi * 0.2**2) < 5e-3, f"disk load should integrate to ~pi r^2, got {area:.4f}"


# ==========================================================================
# region operator converges to the jno.fn-indicator coefficient form
# ==========================================================================
def test_region_operator_converges_to_indicator_reference():
    """``k1*grad over whole + dk*grad over disk`` (piecewise k) converges to ``(k1+dk*ind)*grad``
    under refinement -- they differ only on interface cells (centroid vs quad-point), measure O(h)."""
    diffs = []
    for ms in (0.08, 0.04):
        d, u, phi, (xi, yi), (xb, yb), (ui, vi), (ud, vd) = _disk_setup(mesh_size=ms)
        reg = jno.fem([ui.x * vi.x + ui.y * vi.y, 9.0 * (ud.x * vd.x + ud.y * vd.y), -1.0 * vi, u(xb, yb) - 0.0])
        ind = _indicator(xi, yi)
        ref = jno.fem([(1.0 + 9.0 * ind) * (ui.x * vi.x + ui.y * vi.y), -1.0 * vi, u(xb, yb) - 0.0])
        diffs.append(np.max(np.abs(_solve(reg) - _solve(ref))))
    assert diffs[0] < 5e-3 and diffs[1] < diffs[0], f"region operator should converge to indicator ref, got {diffs}"


def test_two_terms_on_same_region_compose():
    """Two volume terms on the same sub-region (stiffness + a reaction) both restrict correctly --
    region detection and masking are per-term, not per-fem-call."""
    d, u, phi, (xi, yi), (xb, yb), (ui, vi), (ud, vd) = _disk_setup(mesh_size=0.06)
    ind = _indicator(xi, yi)
    reg = jno.fem([ui.x * vi.x + ui.y * vi.y, 3.0 * (ud.x * vd.x + ud.y * vd.y), 5.0 * ud * vd, -1.0 * vi, u(xb, yb) - 0.0])
    ref = jno.fem([(1.0 + 3.0 * ind) * (ui.x * vi.x + ui.y * vi.y), 5.0 * ind * ui * vi, -1.0 * vi, u(xb, yb) - 0.0])
    assert np.max(np.abs(_solve(reg) - _solve(ref))) < 5e-3


# ==========================================================================
# coordinate REUSE across fem() calls keeps the region (the retag-leak guard)
# ==========================================================================
def test_region_survives_coord_reuse_across_fem_calls():
    """`_retag_coords_for_quadrature` rebinds a coord's `.tag` to the quadrature pool in place; reusing
    the SAME sub-region coord objects in a later jno.fem() call must still integrate over the region
    (recovered via `_jno_region_tag`), not silently over the whole domain. Two builds off one stored
    `ud,vd` must give the identical matrix."""
    d, u, phi, (xi, yi), (xb, yb), (ui, vi), (ud, vd) = _disk_setup(mesh_size=0.12)
    A1 = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y, 7.0 * (ud.x * vd.x + ud.y * vd.y), -1.0 * vi, u(xb, yb) - 0.0]).A)
    A2 = np.asarray(jno.fem([ui.x * vi.x + ui.y * vi.y, 7.0 * (ud.x * vd.x + ud.y * vd.y), -1.0 * vi, u(xb, yb) - 0.0]).A)
    # allclose (not array_equal): GPU reductions are not bit-deterministic across two assemblies.
    assert np.allclose(A1, A2), "reusing sub-region coords across fem() calls changed the assembled matrix"
    # and it must NOT have collapsed to the whole-domain operator on the second build
    A_whole = np.asarray(jno.fem([8.0 * (ui.x * vi.x + ui.y * vi.y), -1.0 * vi, u(xb, yb) - 0.0]).A)
    assert not np.allclose(A2, A_whole), "second build silently integrated the region term over the whole domain"


# ==========================================================================
# from_regions geometry parts (interior_<name> tag) restrict the same as a domain.tag predicate
# ==========================================================================
def test_from_regions_per_region_conductivity_restricts():
    """``from_regions`` registers a geometry part's interior under the tag ``interior_<name>`` while the
    part is keyed *bare* in ``_source_regions``. ``_region_and_support`` must map ``interior_<name>`` back
    to the bare region so per-region terms restrict to that part.

    Regression: previously the ``interior_<name>`` tag matched neither ``_source_regions`` (bare keys) nor
    ``_tag_predicates`` (``domain.tag`` only), so the term fell through to whole-domain and every part's
    conductivity was integrated over the entire mesh -- the stiffness silently collapsed to
    ``(sum_p k_p) * K`` and per-material properties had no effect. Two-region series conduction across a
    vertical interface has interface temperature ``kL / (kL + kR)``; the buggy whole-domain assembly gives
    ``0.5`` for every ratio (the scalar ``kL + kR`` cancels)."""
    from shapely.geometry import box

    d = jno.domain.csg.from_regions({"L": box(0, 0, 0.5, 1), "R": box(0.5, 0, 1, 1)}, mesh_size=0.08, time=None)
    pts = np.asarray(d.mesh.points)[:, :2]
    u, v = d.fem_symbols()
    xl, yl, _ = d.variable("interior_L", split=True)
    xr, yr, _ = d.variable("interior_R", split=True)
    ul, vl = u.bind(x=xl, y=yl), v.bind(x=xl, y=yl)
    ur, vr = u.bind(x=xr, y=yr), v.bind(x=xr, y=yr)
    d.tag("xlo", lambda x, y: x < 1e-9)
    d.tag("xhi", lambda x, y: x > 1 - 1e-9)
    xlo, ylo, _ = d.variable("xlo", split=True)
    xhi, yhi, _ = d.variable("xhi", split=True)
    mid = np.abs(pts[:, 0] - 0.5) < 1e-9
    for kL, kR in [(1.0, 100.0), (100.0, 1.0), (1.0, 10.0)]:
        fem = jno.fem(
            [kL * (ul.x * vl.x + ul.y * vl.y), kR * (ur.x * vr.x + ur.y * vr.y), u(xlo, ylo) - 1.0, u(xhi, yhi) - 0.0]
        )
        got = float(_solve(fem)[mid].mean())
        exact = kL / (kL + kR)
        assert abs(got - exact) < 0.03, (
            f"per-region series interface {got:.3f} != kL/(kL+kR)={exact:.3f} (kL={kL}, kR={kR})"
        )


def test_from_regions_multi_region_residual_is_rejected():
    """A single residual that spans two ``from_regions`` parts must raise (the same single-region guard as
    for ``domain.tag`` regions) -- proves the parts are recognized as distinct regions, not silently fused
    into one whole-domain term."""
    from shapely.geometry import box

    d = jno.domain.csg.from_regions({"L": box(0, 0, 0.5, 1), "R": box(0.5, 0, 1, 1)}, mesh_size=0.15, time=None)
    u, v = d.fem_symbols()
    xl, yl, _ = d.variable("interior_L", split=True)
    xr, yr, _ = d.variable("interior_R", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ul, vl = u.bind(x=xl, y=yl), v.bind(x=xl, y=yl)
    ur, vr = u.bind(x=xr, y=yr), v.bind(x=xr, y=yr)
    with pytest.raises(ValueError, match="multiple regions|single region"):
        jno.fem([ul.x * vl.x + ul.y * vl.y + ur.x * vr.x + ur.y * vr.y, u(xb, yb) - 0.0])


# ==========================================================================
# composes with every solve form
# ==========================================================================
def test_region_in_nonlinear_form():
    """A nonlinear reaction ``u^3`` confined to the sub-region matches the jno.fn-indicator form."""
    d, u, phi, (xi, yi), (xb, yb), (ui, vi), (ud, vd) = _disk_setup(mesh_size=0.08)
    ind = _indicator(xi, yi)

    def nl(fem):
        return np.asarray(newton_krylov(lambda w: fem.residual(w), np.full(fem.dofs, 0.1)))

    reg = nl(jno.fem([ui.x * vi.x + ui.y * vi.y, (ud**3) * vd, -1.0 * vi, u(xb, yb) - 0.0]))
    ref = nl(jno.fem([ui.x * vi.x + ui.y * vi.y, ind * (ui**3) * vi, -1.0 * vi, u(xb, yb) - 0.0]))
    assert np.max(np.abs(reg - ref)) < 1e-6


def test_region_in_multifield_form():
    """A sub-region term in one block of a coupled (multi-field) system matches the indicator ref."""
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.08)
    d.tag("disk", _DISK)
    a, qa = d.fem_symbols(names=("a", "qa"))
    b, qb = d.fem_symbols(names=("b", "qb"))
    xi, yi, _ = d.variable("interior", split=True)
    xd, yd, _ = d.variable("disk", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ai, qai = a.bind(x=xi, y=yi), qa.bind(x=xi, y=yi)
    bi, qbi = b.bind(x=xi, y=yi), qb.bind(x=xi, y=yi)
    ad, qad = a.bind(x=xd, y=yd), qa.bind(x=xd, y=yd)
    ind = _indicator(xi, yi)
    coupling = lambda A, qA, B, qB: [B.x * qB.x + B.y * qB.y, A * qB + B * qA, -1.0 * qA - 1.0 * qB]  # noqa: E731
    reg = jno.fem(
        [
            ai.x * qai.x + ai.y * qai.y,
            5.0 * (ad.x * qad.x + ad.y * qad.y),
            *coupling(ai, qai, bi, qbi),
            a(xb, yb) - 0.0,
            b(xb, yb) - 0.0,
        ]
    )
    ref = jno.fem(
        [(1.0 + 5.0 * ind) * (ai.x * qai.x + ai.y * qai.y), *coupling(ai, qai, bi, qbi), a(xb, yb) - 0.0, b(xb, yb) - 0.0]
    )
    assert np.max(np.abs(_solve(reg) - _solve(ref))) < 5e-3


def test_region_in_transient_form():
    """A sub-region stiffness on a transient diffusion matches the jno.fn-indicator trajectory."""
    from jno.utils.solver.backend_blocks import _default_transient_integrate

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1, time=(0.0, 0.05, 6))
    d.tag("disk", _DISK)
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xd, yd, td = d.variable("disk", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, _ = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ud, vd = u.bind(x=xd, y=yd, t=td), phi.bind(x=xd, y=yd, t=td)
    ind = _indicator(xi, yi)
    ts = jnp.linspace(0.0, 0.05, 6)
    reg = jno.fem(
        [ui.t * vi + ui.x * vi.x + ui.y * vi.y, 5.0 * (ud.x * vd.x + ud.y * vd.y), u(xb, yb) - 0.0, u(xi0, yi0) - 1.0]
    )
    ref = jno.fem([ui.t * vi + (1.0 + 5.0 * ind) * (ui.x * vi.x + ui.y * vi.y), u(xb, yb) - 0.0, u(xi0, yi0) - 1.0])
    tr = np.asarray(_default_transient_integrate(reg.operator, {}, ts))
    tf = np.asarray(_default_transient_integrate(ref.operator, {}, ts))
    assert np.max(np.abs(tr - tf)) < 5e-2


def test_region_in_3d():
    """Per-region integration works in 3D (centroid-in-predicate over tetrahedra)."""
    d = jno.domain(constructor=jno.domain.cube(mesh_size=0.18))
    assert d.dimension == 3
    ball = lambda x, y, z: (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 < 0.25**2  # noqa: E731
    d.tag("ball", ball)
    u, phi = d.fem_symbols()
    xi, yi, zi, _ = d.variable("interior", split=True)
    xb, yb, zb, _ = d.variable("boundary", split=True)
    xB, yB, zB, _ = d.variable("ball", split=True)
    ui, vi = u.bind(x=xi, y=yi, z=zi), phi.bind(x=xi, y=yi, z=zi)
    uB, vB = u.bind(x=xB, y=yB, z=zB), phi.bind(x=xB, y=yB, z=zB)
    ind = jno.fn(lambda x, y, z: jnp.where(ball(x, y, z), 1.0, 0.0), [xi, yi, zi])
    reg = jno.fem(
        [
            ui.x * vi.x + ui.y * vi.y + ui.z * vi.z,
            7.0 * (uB.x * vB.x + uB.y * vB.y + uB.z * vB.z),
            -1.0 * vi,
            u(xb, yb, zb) - 0.0,
        ]
    )
    ref = jno.fem([(1.0 + 7.0 * ind) * (ui.x * vi.x + ui.y * vi.y + ui.z * vi.z), -1.0 * vi, u(xb, yb, zb) - 0.0])
    assert np.max(np.abs(_solve(reg) - _solve(ref))) < 5e-3


# ==========================================================================
# train a parameter on a sub-region (the SciML payoff)
# ==========================================================================
def test_scalar_parameter_trained_on_subregion_recovers():
    """A scalar conductivity unknown that multiplies a *sub-region* stiffness is recovered from data
    through ``crux`` -- i.e. you can fit a per-material property on its own subdomain."""
    import optax

    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.1)
    d.tag("disk", _DISK)
    u, phi = d.fem_symbols()
    xi, yi, _ = d.variable("interior", split=True)
    xd, yd, _ = d.variable("disk", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    ui, vi = u.bind(x=xi, y=yi), phi.bind(x=xi, y=yi)
    ud, vd = u.bind(x=xd, y=yd), phi.bind(x=xd, y=yd)
    f = 2.0 * (xi * (1 - xi) + yi * (1 - yi))

    truth = jno.fem([ui.x * vi.x + ui.y * vi.y, 4.0 * (ud.x * vd.x + ud.y * vd.y), -f * vi, u(xb, yb) - 0.0]).solve()

    k = jno.np.parameter((1,), name="kd", key=jax.random.PRNGKey(0))
    k.initialize(jax.nn.initializers.constant(0.5))
    fem = jno.fem([ui.x * vi.x + ui.y * vi.y, k * (ud.x * vd.x + ud.y * vd.y), -f * vi, u(xb, yb) - 0.0])

    crux = jno.core([(fem.solve() - truth).mse], domain=d)
    k.optimizer(optax.adam(3e-1))
    crux.solve(250)
    rec = float(np.asarray(crux.eval([k])).reshape(-1)[0])
    assert abs(rec - 4.0) < 0.3, f"sub-region conductivity should train toward 4.0 (got {rec:.3f})"


# ==========================================================================
# the one not-yet-wired path fails loud (never silently whole-domain)
# ==========================================================================
def test_second_order_in_time_region_fails_loud():
    d = jno.domain(box(0.0, 0.0, 1.0, 1.0), mesh_size=0.3, time=(0.0, 0.1, 5))
    d.tag("disk", _DISK)
    u, phi = d.fem_symbols()
    xi, yi, ti = d.variable("interior", split=True)
    xd, yd, td = d.variable("disk", split=True)
    xb, yb, _ = d.variable("boundary", split=True)
    xi0, yi0, ti0 = d.variable("initial", split=True)
    ui, vi = u.bind(x=xi, y=yi, t=ti), phi.bind(x=xi, y=yi, t=ti)
    ud, vd = u.bind(x=xd, y=yd, t=td), phi.bind(x=xd, y=yd, t=td)
    ui0 = u.bind(x=xi0, y=yi0, t=ti0)
    with pytest.raises(NotImplementedError, match="per-region|second-order"):
        jno.fem(
            [
                ui.tt * vi + ui.x * vi.x + ui.y * vi.y,
                2.0 * (ud.x * vd.x + ud.y * vd.y),
                u(xb, yb) - 0.0,
                u(xi0, yi0) - 1.0,
                ui0.t - 0.0,
            ]
        )
