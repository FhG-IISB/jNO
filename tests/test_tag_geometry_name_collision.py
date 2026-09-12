"""Re-tagging a geometry region with a predicate that means somewhere else must say so.

``Shape.box`` names its own faces (``back bottom front left right top``). ``d.tag(name, pred)`` on
one of those is an ordinary idiom -- ``d.tag("left", lambda x, y: x < 1e-9)`` restates where ``left``
already is -- and 157 call sites in this repo do it.

The trap is that the predicate only filters POINT SAMPLING. Boundary-facet mapping still resolves
through the geometry, so a predicate naming a DIFFERENT face splits the region in two: coordinates
come from the predicate, while a surface term integrates over the geometry's face. Measured on a box
whose ``top`` is ``z=max``, tagged as ``y=max``, with a surface source bound to the tag and a
Dirichlet pinning one face:

    tag name    pin y=max   pin z=max
    a fresh name  0.000 K    28.037 K     <- source really is on y=max
    "top"        25.418 K     0.000 K     <- source is on z=max, the geometry's own face

This is warned, not refused. ``tests/test_fem_history_march.py`` tags a unit cube's ``top`` as
``y=max`` and is CORRECT precisely because of this resolution (it clamps ``z=0`` and wants the
opposite face), so refusing would break working code. Staying silent is how a surface source ends up
on the wrong face of a weld with a perfectly plausible-looking result.
"""

import warnings

import jax
import numpy as np
import pytest

import jno

inner, grad = jno.np.inner, jno.np.grad
LX, LY, LZ, H = 400e-6, 150e-6, 300e-6, 40e-6
RHO, CP, K_TH, T0, Q = 7000.0, 750.0, 30.0, 300.0, 1.0e7


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _box():
    return jno.Shape.box(0.0, 0.0, 0.0, LX, LY, LZ, size=H).domain(time=(0.0, 1e-3, 6))


def test_a_predicate_naming_a_different_face_warns():
    """The melt pool's mistake: a box's `top` is z=max, and tagging it as y=max means somewhere else."""
    d = _box()
    with pytest.warns(UserWarning, match="describes a different part of the boundary"):
        d.tag("top", lambda x, y, z: y > LY - 1e-9)


def test_restating_a_geometry_region_does_not_warn():
    """The common idiom must stay quiet, or the warning is noise and gets filtered out."""
    d = _box()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        d.tag("left", lambda x, y, z: x < 1e-9)


def test_a_name_the_geometry_does_not_own_never_warns():
    """A fresh name has no geometry region to disagree with."""
    d = _box()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d.tag("surface", lambda x, y, z: y > LY - 1e-9)


def test_two_dimensions_where_top_really_is_y_max_does_not_warn():
    """`Shape.rect`'s `top` IS y=max, so the same spelling that warns on a box is right here. This is
    why every 2-D result built on `d.tag("top", y > LY - eps)` is unaffected."""
    d = jno.Shape.rect(0.0, 0.0, 1.0, 1.0, size=0.2).domain()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d.tag("top", lambda x, y: y > 1.0 - 1e-9)


def _rise(tag_name, pin):
    """Heat a box through a surface source bound to `tag_name`; pin one face to T0."""
    d = _box()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the warning is the subject of the tests above
        d.tag(tag_name, lambda x, y, z: y > LY - 1e-9)  # SOURCE: y=max, always
        d.tag("pinned", pin)
    T, S = d.fem_symbols(names=("T", "S"), order=1)
    xi, yi, zi, ti = d.variable("interior", split=True)
    xs, ys, zs, ts_ = d.variable(tag_name, split=True)
    xd, yd, zd, _ = d.variable("pinned", split=True)
    ci = d.variable("initial", split=True)
    ax = [xi, yi, zi]
    Ti, Si = T.bind(x=xi, y=yi, z=zi, t=ti), S.bind(x=xi, y=yi, z=zi, t=ti)
    Ss = S.bind(x=xs, y=ys, z=zs, t=ts_)
    fem = jno.fem(
        [
            RHO * CP * Ti.t * Si + K_TH * inner(grad(T, ax), grad(S, ax), n_contract=1),
            -Q * Ss,
            T(xd, yd, zd) - T0,
            T(*ci) - T0,
        ]
    )
    return float(np.asarray(fem.solve(linear=jno.solve.lu(backend="host")).fn()).max() - T0)


def test_the_warning_is_about_a_real_discrepancy_not_a_style_preference():
    """The teeth. Identical problems, identical source predicate; only the tag NAME differs.

    A fresh name puts the source on y=max, so pinning y=max silences it. The name `top` puts the
    surface integral on the geometry's own face instead, so pinning y=max does NOT silence it and
    pinning z=max does. If this ever stops holding, the warning has become wrong and should go.
    """
    assert _rise("surface", lambda x, y, z: y > LY - 1e-9) < 0.5, (
        "a fresh tag's source is not on y=max -- the premise of the warning is gone"
    )
    assert _rise("surface", lambda x, y, z: z > LZ - 1e-9) > 5.0, "pinning an unrelated face killed the source"
    assert _rise("top", lambda x, y, z: y > LY - 1e-9) > 5.0, (
        "`top` now honours the predicate for facets too -- the discrepancy is fixed and this "
        "warning should be removed rather than kept"
    )
    assert _rise("top", lambda x, y, z: z > LZ - 1e-9) < 0.5, (
        "`top`'s surface integral is no longer on the geometry's own z=max face"
    )
